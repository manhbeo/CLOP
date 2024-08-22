from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset
import os
import pickle
import pytorch_lightning as pl
import torch
import torch.distributed as dist

#TODO: consider iNaturalist
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        file_path = os.path.join(root, 'cifar-10-batches-py', 'data_batch_1' if train else 'test_batch')
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')
        self.fine_labels = self.data['labels']

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
    
class CustomCIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.dataset = datasets.CIFAR100(root=root, train=train, download=True)
        file_path = os.path.join(root, 'cifar-100-python', 'train' if train else 'test')
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')
        self.labels = self.data['coarse_labels']

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


class CustomImageNetDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Check if the dataset is already extracted
        if not os.path.exists(os.path.join(root, split)):
            # If running in distributed mode, only let one process extract
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    self._extract_dataset()
                dist.barrier()  # Wait for rank 0 to finish extracting
            else:
                self._extract_dataset()
        
        self.dataset = datasets.ImageNet(root=root, split=split)
        
        self._setup_labels()

    def _extract_dataset(self):
        # This will trigger the extraction
        datasets.ImageNet(root=self.root, split=self.split, download=True)

    def _setup_labels(self):
        self.labels = self.dataset.targets

    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)

        return (img1, img2), label

    def __len__(self):
        return len(self.dataset)


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data', batch_size=32, dataset="cifar10"):
        super().__init__()
        self.data_dir = data_dir + "_" + dataset
        self.batch_size = batch_size
        self.dataset = dataset
        self.normalize = None

        # Set the correct normalization for the chosen dataset
        #TODO: fix this
        if self.dataset.startswith("cifar"):
            if self.dataset == "cifar10":
                self.normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
            elif self.dataset == "cifar100":
                self.normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                self.normalize
            ])
        elif self.dataset == "imagenet":
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
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
            transforms.Resize(32 if dataset.startswith("cifar") else 256),
            transforms.CenterCrop(32 if dataset.startswith("cifar") else 224),
            transforms.ToTensor(),
            self.normalize
        ])

    def setup(self, stage=None, fraction=1.0):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.dataset == "cifar10":
                full_dataset = CustomCIFAR10Dataset(self.data_dir, train=True, transform=self.train_transform)
            elif self.dataset == "cifar100":
                full_dataset = CustomCIFAR100Dataset(self.data_dir, train=True, transform=self.test_transform)
            elif self.dataset == "imagenet":
                full_dataset = CustomImageNetDataset(self.data_dir, split='train', transform=self.train_transform)

            train_size = int(len(full_dataset) * fraction)
            train_indices = torch.randperm(len(full_dataset))[:train_size]
            train_dataset = Subset(full_dataset, train_indices)

            val_size = int(train_size * 0.1)
            train_size = train_size - val_size

            self.train_dataset, self.val_dataset = random_split(
                train_dataset, [train_size, val_size], generator=torch.Generator()
            )
        if stage == 'test' or stage is None:
            if self.dataset == "cifar10":
                self.test_dataset = CustomCIFAR10Dataset(self.data_dir, train=False, transform=self.test_transform)
            elif self.dataset == "cifar100":
                self.test_dataset = CustomCIFAR100Dataset(self.data_dir, train=False, transform=self.test_transform)
            elif self.dataset == "imagenet":
                self.test_dataset =  CustomImageNetDataset(self.data_dir, split='val', transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True, num_workers=16)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True, num_workers=16)


class CustomEvaluationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32, val_split=0.2, dataset="cifar10"):
        super().__init__()
        self.data_dir = data_dir + "_" + dataset
        self.batch_size = batch_size
        self.dataset = dataset
        self.val_split = val_split

        # Set the correct normalization for the chosen dataset
        if self.dataset == "cifar10":
            self.normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        elif self.dataset == "cifar100":
            self.normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        elif self.dataset == "imagenet":
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Resize(32 if dataset.startswith("cifar") else 256),
            transforms.CenterCrop(32 if dataset.startswith("cifar") else 224),
            transforms.ToTensor(),
            self.normalize
        ])

    def setup(self, stage=None):
        # Split dataset into train, val, and test
        if stage == 'fit' or stage is None:
            if self.dataset == "cifar10":
                full_dataset = CustomCIFAR10Dataset(self.data_dir, train=True, transform=self.train_transform)
            elif self.dataset == "cifar100":
                full_dataset = CustomCIFAR100Dataset(self.data_dir, train=True, transform=self.test_transform)
            elif self.dataset == "imagenet":
                full_dataset = CustomImageNetDataset(self.data_dir, split='train', transform=self.train_transform)

            train_size = int((1 - self.val_split) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.cifar_train, self.cifar_val = random_split(
                full_dataset, [train_size, val_size], generator=torch.Generator()
            )

        if stage == 'test' or stage is None:
            if self.dataset == "cifar10":
                self.test_dataset = CustomCIFAR10Dataset(self.data_dir, train=False, transform=self.test_transform)
            elif self.dataset == "cifar100":
                self.test_dataset = CustomCIFAR100Dataset(self.data_dir, train=False, transform=self.test_transform)
            elif self.dataset == "imagenet":
                self.test_dataset =  CustomImageNetDataset(self.data_dir, split='val', transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, drop_last=True, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=8)