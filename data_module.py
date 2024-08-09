from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import pytorch_lightning as pl
import pickle
import os
import torch
import torch.distributed as dist

#TODO: fix the preprocessing from this file
class CustomCIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.dataset = datasets.CIFAR100(root=root, train=train, download=True)

        # Load the training or testing data
        file_path = os.path.join(root, 'cifar-100-python', 'train' if train else 'test')
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')
        self.fine_labels = self.data['fine_labels']
        self.coarse_labels = self.data['coarse_labels']

    def __getitem__(self, index):
        # Get an image and its fine label
        img, fine_label = self.dataset[index]

        # Transform the image if a transform is provided
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)

        # Get the corresponding coarse label
        coarse_label = self.coarse_labels[index]

        return (img1, img2), fine_label, coarse_label

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
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=32 * 0.1, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def setup(self, stage=None, fraction=1.0):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar100_full = CustomCIFAR100Dataset(self.data_dir, train=True, transform=self.train_transform,)
            train_size = int(len(cifar100_full) * fraction)
            train_indices = torch.randperm(len(cifar100_full))[:train_size]
            train_dataset = Subset(cifar100_full, train_indices)

            val_size = int(train_size*0.1)
            train_size = int(train_size*0.9)

            self.cifar100_train, self.cifar100_val = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar100_test = CustomCIFAR100Dataset(self.data_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=95)

    def val_dataloader(self):
        return DataLoader(self.cifar100_val, batch_size=self.batch_size, drop_last=True, num_workers=95)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size, num_workers=95)


class CIFAR100EvaluationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32, val_split=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])

    def prepare_data(self):
        # Download CIFAR100 dataset
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Split dataset into train, val and test
        if stage == 'fit' or stage is None:
            cifar100_full = datasets.CIFAR100(self.data_dir, train=True, transform=self.transform)
            train_size = int((1 - self.val_split) * len(cifar100_full))
            val_size = len(cifar100_full) - train_size
            self.cifar100_train, self.cifar100_val = random_split(cifar100_full, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.cifar100_test = datasets.CIFAR100(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=95)

    def val_dataloader(self):
        return DataLoader(self.cifar100_val, batch_size=self.batch_size, drop_last=True, num_workers=95)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size, num_workers=95)
    

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
        del temp_dataset  # Free up memory

    def _setup_labels(self):
        label_map = {}
        with open("child_to_parent_labels.txt", 'r') as file:
            for line in file:
                child, parent = line.strip().split('   ')  # 3 spaces
                label_map[child] = parent

        self.label_map = label_map
        self.labels = self.dataset.targets

        self.parent_labels = [int(label_map[str(label)]) for label in self.labels]

    def __getitem__(self, index):
        img, label = self.dataset[index]
        parent_label = self.parent_labels[index]
        
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)

        return (img1, img2), label, parent_label

    def __len__(self):
        return len(self.dataset)


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=31, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None, fraction=1.0):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            imagenet_full = CustomImageNetDataset(self.data_dir, split='train', transform=self.train_transform)
            train_size = int(len(imagenet_full) * fraction)
            train_indices = torch.randperm(len(imagenet_full))[:train_size]
            train_dataset = Subset(imagenet_full, train_indices)

            val_size = int(train_size * 0.1)
            train_size = train_size - val_size

            self.imagenet_train, self.imagenet_val = random_split(train_dataset, [train_size, val_size], generator=torch.Generator())

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.imagenet_test = CustomImageNetDataset(self.data_dir, split='val', transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.imagenet_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.imagenet_val, batch_size=self.batch_size, drop_last=True, num_workers=32)

    def test_dataloader(self):
        return DataLoader(self.imagenet_test, batch_size=self.batch_size, num_workers=32)

class ImageNetEvaluationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=32, val_split=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def prepare_data(self):
        # Download ImageNet dataset
        datasets.ImageNet(self.data_dir, split='train')
        datasets.ImageNet(self.data_dir, split='val')

    def setup(self, stage=None):
        # Split dataset into train, val and test
        if stage == 'fit' or stage is None:
            imagenet_full = datasets.ImageNet(self.data_dir, split='train', transform=self.transform)
            train_size = int((1 - self.val_split) * len(imagenet_full))
            val_size = len(imagenet_full) - train_size
            self.imagenet_train, self.imagenet_val = random_split(imagenet_full, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.imagenet_test = datasets.ImageNet(self.data_dir, split='val', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.imagenet_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.imagenet_val, batch_size=self.batch_size, drop_last=True, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.imagenet_test, batch_size=self.batch_size, num_workers=6)

