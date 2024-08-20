from lightly.models.modules.heads import SimCLRProjectionHead, BarlowTwinsProjectionHead, VICRegProjectionHead
from torchvision import models
import torch.nn as nn
from OARLoss import OARLoss
from lightly.loss.barlow_twins_loss import BarlowTwinsLoss
from lightly.loss.dcl_loss import DCLLoss
from lightly.loss.vicreg_loss import VICRegLoss
from lightly.loss.ntx_ent_loss import NTXentLoss
import pytorch_lightning as pl
import torch
from knn_predict import knn_predict
from LARs import LARS
import math

class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet50_CIFAR, self).__init__()
        self.resnet50 = models.resnet50(weights=None)

        # Modify the initial convolutional layer to better suit CIFAR
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet50.maxpool = nn.Identity()  # Remove the max pooling
        self.resnet50.fc = nn.Identity()  # Remove the final fully connected layer for SimCLR

    def forward(self, x):
        return self.resnet50(x)

class ResNet50_ImgNet(nn.Module):
    def __init__(self):
        super(ResNet50_ImgNet, self).__init__()
        self.resnet50 = models.resnet50(weights=None)
        self.resnet50.fc = nn.Identity()  # Remove the final fully connected layer for SimCLR

    def forward(self, x):
        return self.resnet50(x)

# TODO: consider EMA. do experiment with it 
class CLOA(pl.LightningModule):
    def __init__(self, learning_rate=6.5, lr_schedule="exp", optimizer="lars", criterion="nxt_ent", batch_size=4096, dataset="cifar100", OAR=True, OAR_only=True):
        super(CLOA, self).__init__()

        self.num_classes = 0
        if dataset == "cifar100":
            self.encoder = ResNet50_CIFAR()
            self.num_classes = 100
        if dataset == "cifar10":
            self.encoder = ResNet50_CIFAR()
            self.num_classes = 10
        elif dataset == "imagenet":
            self.encoder = ResNet50_ImgNet()

        if criterion == "nxt_ent":
            if dataset == "cifar100": temperature = 0.5
            else: temperature = 0.1
            self.criterion = NTXentLoss(temperature=temperature, gather_distributed=True)
            self.projection_head = SimCLRProjectionHead(output_dim=128)
            self.output_dim = 128
        elif criterion == "barlow":
            self.criterion = BarlowTwinsLoss(lambda_param=5e-3, gather_distributed=True)
            self.projection_head = BarlowTwinsProjectionHead(output_dim = 8192)
            self.output_dim = 8192
            # batch_size = 256
        elif criterion == "vicreg":
            self.criterion = VICRegLoss(gather_distributed=True)
            self.projection_head = VICRegProjectionHead(output_dim = 8192)
            self.output_dim = 8192
        elif criterion == "dcl":
            self.criterion = DCLLoss(gather_distributed=True)
            self.projection_head = SimCLRProjectionHead(output_dim=128)
            self.output_dim = 128
            # batch_size = 256
        self.OAR = None
        if OAR_only: 
            self.criterion = OARLoss(self.num_classes, embedding_dim=self.output_dim, lambda_value=0.5)
            self.projection_head = SimCLRProjectionHead(output_dim=128)
            self.output_dim = 128
        elif OAR:
            self.OAR = OARLoss(self.num_classes, embedding_dim=self.output_dim, lambda_value=0.5)

        # self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = 0.075 * math.sqrt(batch_size)
        # self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.feature_bank_size = batch_size
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
        assert self.feature_bank_size % batch_size == 0  # for simplicity
        # Replace the features at ptr (oldest features first)
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

        loss = self.criterion(z_i, z_j)
        if self.OAR != None: 
            loss += self.OAR(z_i, fine_label)
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
        k = 100

        pred_labels = knn_predict(z_i, self.feature_bank, self.feature_labels, classes=self.num_classes, knn_k=k, knn_t=0.1)
        correct = (pred_labels == fine_label).sum().item()
        knn_acc = correct / x_i.size(0)
        self.log(f'val_acc', knn_acc, batch_size=x_i.size(0), sync_dist=True)

        loss = self.shared_step(batch)
        self.log('val_loss', loss, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        (x_i, _), _ = batch
        z_i = self.forward(x_i)
        
        # Calculate embedding variance
        z_i_normalized = nn.functional.normalize(z_i, p=2, dim=1)
        embedding_mean = z_i_normalized.mean(dim=0)
        embedding_variance = ((z_i_normalized - embedding_mean) ** 2).mean().item()
        self.log('test_embedding_variance', embedding_variance, batch_size=z_i.size(0), sync_dist=True)

        # Loss calculation
        loss = self.shared_step(batch)
        self.log('test_loss', loss, sync_dist=True)

        return loss

    
    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "lars":
            optimizer = LARS(self.parameters(), lr=self.learning_rate, weight_decay=1e-6)

        if self.lr_schedule == "cosine":    
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000) #TODO: get some warm up
        elif self.lr_schedule == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=500)
        elif self.lr_schedule == "exp":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        scheduler_config = {
            "scheduler": self.scheduler,
            "interval": "epoch",  
            "frequency": 1
        }
        
        return [optimizer], [scheduler_config]
