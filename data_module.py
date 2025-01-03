from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
import pickle
import pytorch_lightning as pl
import torch.distributed as dist
from tinyimagenet import TinyImageNet
from torch.utils.data import Subset
import numpy as np

class CustomCIFAR10Dataset(Dataset):
    '''
        Custom Cifar-10 dataset use for contrastive learning. 
    '''
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        # self.weak_transform = weak_transform
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        file_path = os.path.join(root, 'cifar-10-batches-py', 'data_batch_1' if train else 'test_batch')
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')
        self.fine_labels = self.data['labels']

    def __getitem__(self, index):
        img, fine_label = self.dataset[index]
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
            # img3 = self.transform(img)
            # if self.weak_transform is not None:
            #     img3 = self.weak_transform(img)
        return (img1, img2), fine_label
                
    def __len__(self):
        return len(self.dataset)
    
class CustomCIFAR100Dataset(Dataset):
    '''
        Custom Cifar-100 dataset use for contrastive learning.
    '''
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        # self.weak_transform = weak_transform
        self.dataset = datasets.CIFAR100(root=root, train=train, download=True)
        file_path = os.path.join(root, 'cifar-100-python', 'train' if train else 'test')
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')

    def __getitem__(self, index):
        img, fine_label = self.dataset[index]
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
            # img3 = self.transform(img)
            # if self.weak_transform is not None:
            #     img3 = self.weak_transform(img)
        return (img1, img2), fine_label

    def __len__(self):
        return len(self.dataset)


class CustomImageNetDataset(Dataset):
    '''
        Custom ImageNet and Tiny-ImageNet dataset use for contrastive learning. 
    '''
    def __init__(self, root, split='train', transform=None, dataset="imagenet"):
        self.root = root
        self.split = split
        self.transform = transform
        # self.weak_transform = weak_transform
        
        # Check if the dataset is already extracted
        if not os.path.exists(os.path.join(root, split)):
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    self._extract_dataset(dataset)
                dist.barrier()  # Wait for rank 0 to finish extracting
            else:
                self._extract_dataset(dataset)
        
        if dataset.startswith("tiny"):
            self.dataset = TinyImageNet(root=root, split=split, download=True)
        else:
            self.dataset = datasets.ImageNet(root=root, split=split)
        
        self._setup_labels()

    def _extract_dataset(self, dataset):
        # Trigger the extraction
        if dataset.startswith("tiny"):
            TinyImageNet(root=self.root, split=self.split, download=True)
        else:
            datasets.ImageNet(root=self.root, split=self.split)

    def _setup_labels(self):
        self.labels = self.dataset.targets

    def __getitem__(self, index):
        img, fine_label = self.dataset[index]
        
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
            # img3 = self.transform(img)
            # if self.weak_transform is not None:
            #     img3 = self.weak_transform(img)
        return (img1, img2), fine_label

    def __len__(self):
        return len(self.dataset)


class CustomDataModule(pl.LightningDataModule):
    '''
        Custom datamodule use for training the contrastive learning. 
    '''
    def __init__(self, data_dir='data', batch_size=32, dataset="cifar100", num_workers=9, augment="rand", loss="supcon"):
        '''
           Parameters:
            - data_dir (str): The base directory where the dataset is located. The final directory will be a combination of this and the dataset name.
            - batch_size (int): The size of the data batches to be loaded.
            - dataset (str): The name of the dataset to be used ('cifar100', 'cifar10', 'tiny_imagenet', 'imagenet'). 
            - num_workers (int): The number of subprocesses used for data loading.
            - augment (str): The type of data augmentation to be applied('rand' for RandAugment, 'auto' for ImageNet AutoAugment, 'sim' for SimCLR). 
                             Works only when training on Tiny-ImageNet.
            - loss (str): The loss function to be used ('ntx_ent' for unsupervised contrastive loss, 'supcon' for supervised contrastive loss). 
        '''
        super().__init__()
        self.data_dir = data_dir + "_" + dataset
        self.batch_size = batch_size
        self.dataset = dataset
        self.loss = loss

        if self.dataset.startswith("cifar"):
            resize_size = 32
            crop_size = 32
            if self.dataset == "cifar10":
                normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
            elif self.dataset == "cifar100":
                normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        elif self.dataset == "imagenet":
            resize_size = 256
            crop_size = 224
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
            ])

        elif self.dataset == "tiny_imagenet":
            resize_size = 64
            crop_size = 64
            normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
            if augment == "sim":
                self.train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=64),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
                    transforms.ToTensor(),
                    normalize
                ])

            elif augment == "auto":
                self.train_transform = transforms.Compose([
                    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
                    transforms.ToTensor(),
                    normalize
                ])
            elif augment == "rand":
                self.train_transform = transforms.Compose([
                    transforms.RandAugment(num_ops=3, magnitude=18),
                    transforms.ToTensor(),
                    normalize
                ])
        # if self.loss != "supcon":
        #     self.weak_transform = transforms.Compose([
        #         transforms.Resize(resize_size),
        #         transforms.CenterCrop(crop_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
        #         transforms.ToTensor(),
        #         normalize
        #     ])
        # else: self.weak_transform = None

        self.val_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])
        self.num_workers = num_workers

    def setup(self, stage):
        if self.dataset == "cifar10":
            self.train_dataset = CustomCIFAR10Dataset(self.data_dir, train=True, transform=self.train_transform)
            self.val_dataset = CustomCIFAR10Dataset(self.data_dir, train=False, transform=self.val_transform)
        elif self.dataset == "cifar100":
            self.train_dataset = CustomCIFAR100Dataset(self.data_dir, train=True, transform=self.train_transform)
            self.val_dataset = CustomCIFAR100Dataset(self.data_dir, train=False, transform=self.val_transform)
        elif self.dataset == "imagenet":
            self.train_dataset = CustomImageNetDataset(self.data_dir, split='train', transform=self.train_transform, dataset="imagenet")
            self.val_dataset =  CustomImageNetDataset(self.data_dir, split='val', transform=self.val_transform, dataset="imagenet")
        elif self.dataset == "tiny_imagenet":
            self.train_dataset = CustomImageNetDataset(self.data_dir, split='train', transform=self.train_transform, dataset="tiny_imagenet")
            self.val_dataset =  CustomImageNetDataset(self.data_dir, split='val', transform=self.val_transform, dataset="tiny_imagenet")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class CustomEvaluationDataModule(pl.LightningDataModule):
    '''
        Custom datamodule use for training the linear classifier. 
    '''
    def __init__(self, data_dir='./data', batch_size=32, dataset="cifar100", num_workers=9, augment="rand", subset_fraction=1.0):
        '''
        Parameters:
        - data_dir (str): The base directory where the dataset is located. The final directory will be a combination of this and the dataset name. 
        - batch_size (int): The number of data samples in each batch during data loading.
        - dataset (str): The name of the dataset to be used ('cifar100', 'cifar10', 'tiny_imagenet', 'imagenet').
        - num_workers (int): The number of worker processes used for data loading. 
        - augment (str): The type of data augmentation to be applied('rand' for RandAugment, 'auto' for ImageNet AutoAugment, 'sim' for SimCLR). 
                         Works only when training on Tiny-ImageNet.
        - subset_fraction (float): A fraction (0.0 to 1.0) of the dataset to be used to train the linear classifier.
        '''
        super().__init__()
        self.data_dir = data_dir + "_" + dataset
        self.batch_size = batch_size
        self.dataset = dataset
        self.subset_fraction = subset_fraction

        if self.dataset.startswith("cifar"):
            resize_size = 32
            crop_size = 32
            if self.dataset == "cifar10":
                normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
            elif self.dataset == "cifar100":
                normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        elif self.dataset == "imagenet":
            resize_size = 256
            crop_size = 224
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                normalize
            ])

        elif self.dataset == "tiny_imagenet":
            resize_size = 64
            crop_size = 64
            normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
            if augment == "sim":
                self.train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=64),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
                    transforms.RandomSolarize(threshold=128, p=0.1),
                    transforms.ToTensor(),
                    normalize
                ])

            elif augment == "auto":
                self.train_transform = transforms.Compose([
                    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
                    transforms.ToTensor(),
                    normalize
                ])
            elif augment == "rand":
                self.train_transform = transforms.Compose([
                    transforms.RandAugment(num_ops=3, magnitude=18),
                    transforms.ToTensor(),
                    normalize
                ])
            elif augment == "val":
                self.train_transform = transforms.Compose([
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    normalize
                ])

        self.val_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])
        self.num_workers = num_workers

    def setup(self, stage):
        if self.dataset == "cifar10":
            train_dataset = datasets.CIFAR10(self.data_dir, train=True, transform=self.train_transform, download=True)
            self.val_dataset = datasets.CIFAR10(self.data_dir, train=False, transform=self.val_transform, download=True)
        elif self.dataset == "cifar100":
            train_dataset = datasets.CIFAR100(self.data_dir, train=True, transform=self.train_transform, download=True)
            self.val_dataset = datasets.CIFAR100(self.data_dir, train=False, transform=self.val_transform, download=True)
        elif self.dataset == "imagenet":
            train_dataset = datasets.ImageNet(self.data_dir, split='train', transform=self.train_transform)
            self.val_dataset =  datasets.ImageNet(self.data_dir, split='val', transform=self.val_transform)
        elif self.dataset == "tiny_imagenet":
            train_dataset = TinyImageNet(self.data_dir, split='train', transform=self.train_transform, download=True)
            self.val_dataset =  TinyImageNet(self.data_dir, split='val', transform=self.val_transform, download=True)

        if self.subset_fraction < 1.0:
            num_samples = int(len(train_dataset) * self.subset_fraction)
            indices = np.random.choice(len(train_dataset), num_samples, replace=False)
            self.train_dataset = Subset(train_dataset, indices)
        else:
            self.train_dataset = train_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    