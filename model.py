from lightly.models.modules.heads import SimCLRProjectionHead
from torchvision import models
import torch.nn as nn
from OARLoss import OARLoss
from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.loss.barlow_twins_loss import BarlowTwinsLoss
from supervised import Supervised_NTXentLoss
import pytorch_lightning as pl
import torch
from knn_predict import knn_predict
from LARs import LARS
import math
from lightly.utils.scheduler import CosineWarmupScheduler
import torch.nn.functional as F

class ResNet50_cifar(nn.Module):
    def __init__(self):
        super(ResNet50_cifar, self).__init__()
        self.resnet50 = models.resnet50(weights=None)

        # Modify the initial convolutional layer to better suit CIFAR
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet50.maxpool = nn.Identity()  # Remove the max pooling
        self.resnet50.fc = nn.Identity()  # Remove the final fully connected layer for SimCLR

    def forward(self, x):
        x = self.resnet50(x)
        return F.normalize(x, dim=1)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(weights=None)
        self.resnet50.fc = nn.Identity()  # Remove the final fully connected layer for SimCLR

    def forward(self, x):
        x = self.resnet50(x)
        return F.normalize(x, dim=1)

# TODO: consider EMA. do experiment with it 
class CLOA(pl.LightningModule):
    def __init__(self, batch_size=128, dataset="tiny_imagenet", OAR=True, loss="supcon", devices=1, k=100, distance="cosine", learning_rate=None, lambda_val=1.0):
        super(CLOA, self).__init__()
        self.dataset = dataset
        self.k = k

        if dataset.startswith("cifar") or dataset == "tiny_imagenet":
            self.encoder = ResNet50_cifar()
            if dataset == "cifar10": 
                self.num_classes = 10
            elif dataset == "cifar100": 
                self.num_classes = 100
            elif dataset == "tiny_imagenet":
                self.num_classes = 200
        elif dataset == "imagenet":
            self.encoder = ResNet50()
            self.num_classes = 1000    

        self.loss = loss
        if self.loss == "ntx_ent" or self.loss == "supcon":
            if dataset == "cifar10": 
                temperature = 0.5
                self.output_dim = 128
            elif dataset == "cifar100": 
                temperature = 0.2
                self.output_dim = 128
            elif dataset == "imagenet":
                temperature = 0.1
                self.output_dim = 1024
            elif dataset == "tiny_imagenet":
                temperature = 0.1
                self.output_dim = 256
            if self.loss == "nxt_ent":
                self.criterion = NTXentLoss(temperature=temperature, gather_distributed=True)
            elif self.loss == "supcon":
                self.criterion = Supervised_NTXentLoss(temperature=temperature, gather_distributed=True)
        elif self.loss == "barlow":
            self.criterion = BarlowTwinsLoss(gather_distributed=True)

        self.OAR = None    
        if OAR:
            self.OAR = OARLoss(self.num_classes, embedding_dim=self.output_dim, lambda_value=lambda_val, distance=distance)

        self.projection_head = SimCLRProjectionHead(output_dim=self.output_dim)

        if learning_rate is None:
            if dataset.startswith("cifar"):
                self.learning_rate = 0.075 * math.sqrt(batch_size*devices)
            else:
                self.learning_rate = 0.3 * (batch_size*devices) / 256
        else:
            self.learning_rate = learning_rate
        self.feature_bank_size = 2048
        self._init_feature_bank(self.feature_bank_size)
    
    def _init_feature_bank(self, feature_bank_size):
        # Initialize feature bank and labels
        self.feature_bank_size = feature_bank_size
        self.register_buffer("feature_bank", torch.randn(feature_bank_size, self.output_dim))
        self.register_buffer("feature_labels", torch.randint(0, self.num_classes, (feature_bank_size,)))
        self.feature_bank_ptr = 0

    def _update_feature_bank(self, features, labels):
        batch_size = features.size(0)
        ptr = self.feature_bank_ptr
        assert self.feature_bank_size % batch_size == 0
        self.feature_bank[ptr:ptr + batch_size, :] = features.detach()
        self.feature_labels[ptr:ptr + batch_size] = labels.detach()
        # Move the pointer
        self.feature_bank_ptr = (self.feature_bank_ptr + batch_size) % self.feature_bank_size

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z

    def shared_step(self, batch):
        (x_i, x_j), fine_label = batch
        z_i = self.forward(x_i)
        z_j = self.forward(x_j)

        if self.loss == "supcon": 
            loss = self.criterion(z_i, z_j, fine_label)
        else:    
            loss = self.criterion(z_i, z_j)
        if self.OAR != None: 
            loss += self.OAR(z_i, z_j, fine_label)
        return loss

    def training_step(self, batch, batch_idx):
        (x_i, _), fine_label = batch
        z_i = self.forward(x_i)
        loss = self.shared_step(batch)
        self.log('train_loss', loss, sync_dist=True)
        self._update_feature_bank(z_i, fine_label)
        return loss

    def validation_step(self, batch, batch_idx):
        (x_i, _), fine_label = batch
        z_i = self.forward(x_i)
        k = self.k

        pred_labels = knn_predict(z_i, self.feature_bank, self.feature_labels, classes=self.num_classes, knn_k=k, knn_t=0.1)
        correct = (pred_labels == fine_label).sum().item()
        knn_acc = correct / x_i.size(0)
        self.log(f'knn_acc-k={k}', knn_acc, batch_size=x_i.size(0), sync_dist=True)

        loss = self.shared_step(batch)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        (x_i, _), _ = batch
        z_i = self.forward(x_i)

        # Calculate embedding variance
        z_i_normalized = nn.functional.normalize(z_i, p=2, dim=1)
        embedding_variance = torch.var(z_i_normalized, dim=0).mean().item()
        self.log('test_embedding_variance', embedding_variance, sync_dist=True)
        return embedding_variance

    
    def configure_optimizers(self):
        optimizer = LARS(self.parameters(), lr=self.learning_rate, weight_decay=1e-6)
        self.scheduler = CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0 if self.dataset.startswith("cifar") else 10,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            )

        scheduler_config = {
            "scheduler": self.scheduler,
            "interval": "epoch",  
            "frequency": 1
        }
        
        return [optimizer], [scheduler_config]
