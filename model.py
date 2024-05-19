from lightly.models.modules.heads import SimCLRProjectionHead
from torchvision import models
import torch.nn as nn
from losses import SupConLoss, OARLoss, SupervisedVICRegLoss
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.optim as optim
from lars import LARS


class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18_CIFAR, self).__init__()
        # Load a pre-initialized ResNet-18 model without pretrained weights
        self.resnet18 = models.resnet18(weights=None)

        # Modify the initial convolutional layer to better suit CIFAR
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()  # Remove the max pooling
        self.resnet18.fc = nn.Identity()  # Remove the final fully connected layer for SimCLR

    def forward(self, x):
        return self.resnet18(x)

class TreeCLR(pl.LightningModule):
    def __init__(self, learning_rate, optimizer, lr_schedule, temperature, ):
        super(TreeCLR, self).__init__()

        self.encoder = ResNet18_CIFAR()
        self.projection_head = SimCLRProjectionHead(input_dim=512, hidden_dim=256, output_dim=128)
        
        self.criterion = SupTreeConLoss(temperature=temperature, use_distributed=True)
        
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate

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
        loss = self.shared_step(batch)
        self.log('treeCLR_train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('treeCLR_val_loss', loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'adam':
          optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'lars':
          optimizer = LARS(self.parameters(), lr=self.learning_rate, momentum=0.9)
          
        if self.lr_schedule == "linear":
          scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
          scheduler_config = {"scheduler": scheduler, "interval": "epoch"}
        elif self.lr_schedule == "cosine":
          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
          scheduler_config = {"scheduler": scheduler, "interval": "epoch"}
        
        return [optimizer], [scheduler_config]
        
    def total_steps(self):
        # Calculate the total training steps required for OneCycleLR
        num_epochs = self.trainer.max_epochs
        num_train_batches = len(self.train_dataloader())
        return num_epochs * num_train_batches

class LinearClassifierModule(pl.LightningModule):
    def __init__(self, encoder, feature_dim=512, num_classes=100, learning_rate=0.01):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)


        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            representations = self.encoder(x)
        return self.classifier(representations)

    def training_step(self, batch, batch_idx):
        (x_i, _), labels, _ = batch
        logits = self(x_i)
        loss = self.criterion(logits, labels)
        acc = self.train_accuracy(logits.softmax(dim=-1), labels)
        self.log('eval_train_loss', loss)
        self.log('eval_train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x_i, _), labels, _ = batch
        logits = self(x_i)
        loss = self.criterion(logits, labels)
        acc = self.val_accuracy(logits.softmax(dim=-1), labels)
        self.log('eval_val_loss', loss)
        self.log('eval_val_acc', acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate, weight_decay=1e-6)
        return optimizer

    def test_step(self, batch, batch_idx):
        (x_i, _), labels, _ = batch
        logits = self(x_i)
        loss = self.criterion(logits, labels)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_accuracy(logits, labels))
        return loss
