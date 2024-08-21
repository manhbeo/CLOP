from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset
import os
import pickle
import pytorch_lightning as pl
import torch

#TODO: do iNaturalist and ImgNet
class CustomDataset(Dataset):
    def __init__(self, root, dataset, train=True, transform=None):
        self.transform = transform
        if dataset == "cifar100": 
            self.dataset = datasets.CIFAR100(root=root, train=train, download=True)
            file_path = os.path.join(root, 'cifar-100-python', 'train' if train else 'test')
            with open(file_path, 'rb') as f:
                self.data = pickle.load(f, encoding='latin1')
            self.fine_labels = self.data['fine_labels']

        if dataset == "imagenet": 
            subdir = 'train' if train else 'val'
            self.dataset = datasets.ImageFolder(root=os.path.join(root, subdir), transform=self.transform)

    def __getitem__(self, index):
        # Get an image and its fine label
        img, fine_label = self.dataset[index]

        # Transform the image if a transform is provided
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        else:
            img1 = img
            img2 = img

        return (img1, img2), fine_label

    def __len__(self):
        return len(self.dataset)


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data', batch_size=32, dataset="cifar100"):
        super().__init__()
        self.data_dir = data_dir + "_" + dataset
        self.batch_size = batch_size
        self.dataset = dataset
        self.normalize = None

        # Set the correct normalization for the chosen dataset
        if self.dataset == "cifar100":
            self.normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
            crop_size = 32
        elif self.dataset == "imagenet":
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            crop_size = 224

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(32 * 0.1), sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            self.normalize
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(32 if self.dataset == "cifar100" else 256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            self.normalize
        ])

    def setup(self, stage=None, fraction=1.0):
        # Assign train/val datasets for use in dataloaders
        cifar_full = CustomDataset(self.data_dir, train=True, transform=self.train_transform)

        train_size = int(len(cifar_full) * fraction)
        train_indices = torch.randperm(len(cifar_full))[:train_size]
        train_dataset = Subset(cifar_full, train_indices)

        val_size = int(train_size * 0.1)
        train_size = int(train_size * 0.9)

        self.cifar_train, self.cifar_val = random_split(
            train_dataset, [train_size, val_size], generator=torch.Generator()
        )

        self.cifar_test = CustomDataset(self.data_dir, self.dataset, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, drop_last=True, num_workers=16)
    
    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, drop_last=True, num_workers=16)


class CIFAREvaluationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32, val_split=0.2, dataset="cifar100"):
        super().__init__()
        self.data_dir = data_dir + "_" + dataset
        self.batch_size = batch_size
        self.dataset = dataset
        self.val_split = val_split

        # Set the correct normalization for the chosen dataset
        if self.dataset == "cifar100":
            self.normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
            crop_size = 32
        elif self.dataset == "imagenet":
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            crop_size = 224

        self.transform = transforms.Compose([
            transforms.Resize(32 if dataset == "cifar100" else 256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            self.normalize
        ])

    def setup(self, stage=None):
        # Split dataset into train, val, and test
        if stage == 'fit' or stage is None:
            cifar_full = CustomDataset(self.data_dir, self.dataset, train=True, transform=self.train_transform)

            train_size = int((1 - self.val_split) * len(cifar_full))
            val_size = len(cifar_full) - train_size
            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [train_size, val_size], generator=torch.Generator()
            )

        if stage == 'test' or stage is None:
            self.cifar_test = CustomDataset(self.data_dir, self.dataset, train=False, transform=self.train_transform)


    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, drop_last=True, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=8)