import os
import pickle
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import torch
import pytorch_lightning as pl
from torchvision import transforms, datasets

class CustomCIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.dataset = datasets.CIFAR100(root=root, train=train, download=True)

    def __getitem__(self, index):
        # Get an image and its label
        img, label = self.dataset[index]

        # Apply the transform to generate two views of the same image if a transform is provided
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)

        return (img1, img2), label

    def __len__(self):
        return len(self.dataset)

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(32 * 0.1), sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_train_dataset = CustomCIFAR100Dataset(self.data_dir, train=True, transform=self.train_transform)
            train_size = int(0.9 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        
        if stage == 'test' or stage is None:
            self.test_dataset = CustomCIFAR100Dataset(self.data_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)