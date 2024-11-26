from lightly.models.modules.heads import SimCLRProjectionHead
from torchvision import models
import torch.nn as nn
from CLOPLoss import CLOPLoss
from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.loss.vicreg_loss import VICRegLoss
from supervised import Supervised_NTXentLoss
import pytorch_lightning as pl
import torch
from knn_predict import knn_predict
from LARs import LARS
import math
from lightly.utils.scheduler import CosineWarmupScheduler
import torch.nn.functional as F

class ResNet50_small(nn.Module):
    '''
        ResNet50 with modification to suit contrastive learning on Cifar and Tiny-ImageNet
    '''
    def __init__(self):
        super(ResNet50_small, self).__init__()
        self.resnet50 = models.resnet50(weights=None)
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet50.maxpool = nn.Identity()  
        self.resnet50.fc = nn.Identity()  
    def forward(self, x):
        x = self.resnet50(x)
        return F.normalize(x, dim=1)

class ResNet50(nn.Module):
    '''
        ResNet50 with modification to suit contrastive learning on ImageNet
    '''
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(weights=None)
        self.resnet50.fc = nn.Identity()  
    def forward(self, x):
        x = self.resnet50(x)
        return F.normalize(x, dim=1)

class CLOP(pl.LightningModule):
    def __init__(self, batch_size=128, dataset="cifar100", has_CLOP=True, loss="supcon", devices=1, k=100, distance="cosine",
                 learning_rate=None, lambda_val=1.0, label_por=1.0, etf=False, semi=False):
        '''
        Parameters:
        - batch_size (int): Batch size
        - dataset (str): The name of the dataset to be used ('cifar100', 'cifar10', 'tiny_imagenet', 'imagenet'). 
        - has_CLOP (bool): A boolean flag indicating whether to use the CLOP loss.
        - loss (str): The loss function to be used ('ntx_ent' for unsupervised contrastive loss, 'supcon' for supervised contrastive loss). 
        - devices (int): The number of GPUs to be used. 
        - k (int): The number of nearest neighbors for k-NN accuracy validation
        - distance (str): The distance metric to be used 
                          ('cosine' for cosine similarity, 'euclidean' for Euclidean distance, "manhattan" for Manhattan distance)
        - learning_rate (float or None): The learning rate for the optimizer. 
                                         If None, a default learning rate of 0.3 * (batch_size/256) is used. 
        - lambda_val (float): The weighting factor for CLOP loss.
        - label_por (float): The proportion of labeled data to be used on contrastive learning.
        '''
        super(CLOP, self).__init__()
        self.dataset = dataset
        self.k = k

        if dataset.startswith("cifar") or dataset == "tiny_imagenet":
            self.encoder = ResNet50_small()
            if dataset == "cifar10": 
                self.num_classes = 10
                self.output_dim = 128
            elif dataset == "cifar100": 
                self.num_classes = 100
                self.output_dim = 128
            elif dataset == "tiny_imagenet":
                self.num_classes = 200
                self.output_dim = 256
        elif dataset == "imagenet":
            self.encoder = ResNet50()
            self.num_classes = 1000    
            self.output_dim = 1024

        self.loss = loss
        self.semi_loss = None
        if self.loss != "CLOP_only":
            if dataset == "cifar10": 
                temperature = 0.5
            elif dataset == "tiny_imagenet":
                temperature = 0.1
            elif dataset == "cifar100": 
                temperature = 0.2
            elif dataset == "imagenet":
                temperature = 0.1
            if self.loss == "ntx_ent":
                self.criterion = NTXentLoss(temperature=temperature, gather_distributed=True)
            elif self.loss == "supcon":
                self.criterion = Supervised_NTXentLoss(temperature=temperature, label_fraction=label_por, gather_distributed=True)
            elif self.loss == "vicreg":
                self.criterion = VICRegLoss(gather_distributed=True)
                if semi:
                    self.semi_loss = Supervised_NTXentLoss(temperature=temperature, label_fraction=label_por, gather_distributed=True)
        else:
            self.criterion = CLOPLoss(self.num_classes, self.output_dim, lambda_val, distance, label_por, etf)

        self.has_CLOP = None    
        if has_CLOP:
            self.CLOPLoss = CLOPLoss(self.num_classes, embedding_dim=self.output_dim, lambda_value=lambda_val, distance=distance, label_por=label_por)

        self.projection_head = SimCLRProjectionHead(output_dim=self.output_dim)

        if learning_rate is None:
            if dataset.startswith("cifar"):
                if self.loss == "supcon":
                    self.learning_rate = 0.075 * math.sqrt(batch_size*devices)
                else:
                    self.learning_rate = 0.3 * (batch_size*devices) / 256
            elif dataset == "tiny_imagenet":
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

    def shared_step(self, batch, label_por=None):
        (x_i, x_j), fine_label = batch
        z_i = self.forward(x_i)
        z_j = self.forward(x_j)

        if self.loss == "supcon": 
            loss = self.criterion(z_i, z_j, fine_label)
            if self.has_CLOP != None: 
                loss += self.CLOPLoss(z_i, z_j, None, fine_label, label_por, None)
        elif self.loss == "vicreg":
            loss = self.criterion(z_i, z_j)
            if self.semi_loss is not None:
                loss += self.semi_loss(z_i, z_j, fine_label)
        else:    
            loss = self.criterion(z_i, z_j)
            if self.has_CLOP != None: 
                # z_weak = self.forward(x_weak)
                loss += self.CLOPLoss(z_i, z_j, None, current_epoch=self.current_epoch)
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

        loss = self.shared_step(batch, 1.0)
        self.log('val_loss', loss, sync_dist=True)
        return loss
    
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
