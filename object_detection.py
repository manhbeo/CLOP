import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class ObjectDetectionHead(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        # Create Faster R-CNN with pretrained backbone
        self.detector = fasterrcnn_resnet50_fpn(pretrained=False)

        # Replace the backbone with our pretrained model
        self.detector.backbone.body = backbone

        # Replace classification head
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.detector(images, targets)


class ObjectDetectionClassifier(pl.LightningModule):
    def __init__(self, pretrained_model, batch_size, num_classes, freeze_backbone=True, lr=None):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_model'])

        # Remove projection head and get backbone
        backbone = pretrained_model.backbone
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.model = ObjectDetectionHead(backbone, num_classes)
        self.lr = lr or 1e-4

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Log each loss component
        for k, v in loss_dict.items():
            self.log(f'train_{k}', v, on_step=True, on_epoch=True, prog_bar=True)

        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Log each loss component
        for k, v in loss_dict.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, prog_bar=True)

        return losses

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)