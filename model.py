from lightly.models.modules.heads import SimCLRProjectionHead
from torchvision import models
import torch.nn as nn
from SupTreeConLoss import SupTreeConLoss 
# adding some losses here
from OARLoss import OARLoss
import pytorch_lightning as pl
import torch
from knn_predict import knn_predict
from LARs import LARS

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

# TODO: consider EMA
class TreeCLR(pl.LightningModule):
    def __init__(self, learning_rate=1.2, lr_schedule="exp", optimizer="lars", temperature=0.192, lambda_val=0.5, feature_bank_size=128, dataset = "imagenet", have_coarse_label=True):
        super(TreeCLR, self).__init__()

        if dataset == "cifar100":
            self.encoder = ResNet50_CIFAR()
        elif dataset == "imagenet":
            self.encoder = ResNet50_ImgNet()
        self.projection_head = SimCLRProjectionHead(input_dim=2048, hidden_dim=256, output_dim=128)
        
        self.criterion = SupTreeConLoss(temperature=temperature, lambda_val=lambda_val, use_distributed=True)

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.have_coarse_label = have_coarse_label
        self.feature_bank_size = feature_bank_size
        self._init_feature_bank(self.feature_bank_size)
    
    def _init_feature_bank(self, feature_bank_size):
        # Initialize feature bank and labels
        self.feature_bank_size = feature_bank_size
        self.register_buffer("feature_bank", torch.randn(feature_bank_size, 128))
        self.register_buffer("feature_labels", torch.randint(0, 100, (feature_bank_size,)))
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
        (x_i, x_j), fine_label, coarse_label = batch
        if not self.have_coarse_label: 
          coarse_label = None
        z_i = self.forward(x_i)
        z_j = self.forward(x_j)

        loss = self.criterion(z_i, z_j, fine_label, coarse_label)
        return loss

    def training_step(self, batch, batch_idx):
        (x_i, _), fine_label, _ = batch
        z_i = self.forward(x_i)
        loss = self.shared_step(batch)
        self.log('train_loss', loss, sync_dist=True)
        self._update_feature_bank(z_i, fine_label)
        return loss

    def validation_step(self, batch, batch_idx):
        (x_i, _), fine_label, _ = batch
        z_i = self.forward(x_i)
        k = 100

        pred_labels = knn_predict(z_i, self.feature_bank, self.feature_labels, classes=100, knn_k=k, knn_t=0.1)
        correct = (pred_labels == fine_label).sum().item()
        knn_acc = correct / x_i.size(0)
        self.log(f'val_acc', knn_acc, batch_size=x_i.size(0), sync_dist=True)

        loss = self.shared_step(batch)
        self.log('val_loss', loss, sync_dist=True)
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
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        elif self.lr_schedule == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=500)
        elif self.lr_schedule == "exp":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        scheduler_config = {
            "scheduler": self.scheduler,
            "interval": "epoch",  
            "frequency": 1
        }
        
        return [optimizer], [scheduler_config]
