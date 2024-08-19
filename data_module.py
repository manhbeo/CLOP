from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import pytorch_lightning as pl
import torch

class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data', batch_size=32, dataset="cifar100"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset = dataset

        # Set the correct normalization for the chosen dataset
        if self.dataset == "cifar100":
            self.normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        elif self.dataset == "cifar10":
            self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(32 * 0.1), sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            self.normalize
        ])

    def setup(self, stage=None, fraction=1.0):
        # Assign train/val datasets for use in dataloaders
        if self.dataset == "cifar100":
            cifar_full = datasets.CIFAR100(self.data_dir, train=True, transform=self.train_transform, download=True)
        elif self.dataset == "cifar10":
            cifar_full = datasets.CIFAR10(self.data_dir, train=True, transform=self.train_transform, download=True)

        train_size = int(len(cifar_full) * fraction)
        train_indices = torch.randperm(len(cifar_full))[:train_size]
        train_dataset = Subset(cifar_full, train_indices)

        val_size = int(train_size * 0.1)
        train_size = int(train_size * 0.9)

        self.cifar_train, self.cifar_val = random_split(
            train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=95)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, drop_last=True, num_workers=95)


class CIFAREvaluationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32, val_split=0.2, dataset="cifar100"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset = dataset
        self.val_split = val_split

        # Set the correct normalization for the chosen dataset
        if self.dataset == "cifar100":
            self.normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        elif self.dataset == "cifar10":
            self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

    def setup(self, stage=None):
        # Split dataset into train, val, and test
        if stage == 'fit' or stage is None:
            if self.dataset == "cifar100":
                cifar_full = datasets.CIFAR100(self.data_dir, train=True, transform=self.transform, download=True)
            elif self.dataset == "cifar10":
                cifar_full = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform, download=True)

            train_size = int((1 - self.val_split) * len(cifar_full))
            val_size = len(cifar_full) - train_size
            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )

        if stage == 'test' or stage is None:
            if self.dataset == "cifar100":
                self.cifar_test = datasets.CIFAR100(self.data_dir, train=False, transform=self.transform, download=True)
            elif self.dataset == "cifar10":
                self.cifar_test = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, drop_last=True, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=8)