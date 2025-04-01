from torch.utils.data import DataLoader, Dataset
import os
import torch.distributed as dist
from tinyimagenet import TinyImageNet
from typing import Dict, Tuple
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Literal, Optional
random.seed(42)

class CustomCIFAR100Dataset(Dataset):
    '''
    Custom CIFAR-100 dataset with class-splitting and label percentage support for contrastive learning.
    '''

    def __init__(self, root, train=True, transform=None,
                 class_percentages: Optional[Dict[int, float]] = None,
                 label_percentage: float = 1.0):
        """
        Args:
            root: Root directory of the dataset
            train: If True, creates dataset from training set, otherwise creates from test set
            transform: Data transformations
            class_percentages: Dictionary mapping class indices to percentage of samples to keep (0.0 to 1.0)
                             e.g., {0: 0.5, 1: 1.0} keeps 50% of class 0 and 100% of class 1
            label_percentage: Percentage of samples that will retain their labels (0.0 to 1.0)
                            The rest will have their labels masked with -1
        """
        self.transform = transform
        self.dataset = datasets.CIFAR100(root=root, train=train, download=True)
        self.class_percentages = class_percentages
        self.label_percentage = label_percentage

        self._setup_labels()

        if self.class_percentages is not None:
            self._apply_class_splitting()

        if self.label_percentage < 1.0:
            self._apply_label_masking()

    def _setup_labels(self):
        """Initialize labels and sample indices"""
        self.labels = self.dataset.targets
        self.samples_idx = list(range(len(self.dataset)))
        self.masked_labels = self.labels.copy()  # Create a copy for label masking

    def _apply_class_splitting(self):
        """Apply class-splitting based on specified percentages"""
        # Group indices by class
        class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Apply splitting and create final indices list
        final_indices = []
        for class_idx, indices in class_indices.items():
            percentage = self.class_percentages.get(class_idx, 1.0)  # Default to 100% if not specified
            num_samples = int(len(indices) * percentage)
            if num_samples > 0:
                selected_indices = random.sample(indices, num_samples)
                final_indices.extend(selected_indices)

        # Update samples index list
        self.samples_idx = sorted(final_indices)

    def _apply_label_masking(self):
        """Mask labels for a percentage of samples"""
        num_samples = len(self.samples_idx)
        num_labeled = int(num_samples * self.label_percentage)

        # Randomly select indices to keep labels
        labeled_indices = set(random.sample(range(num_samples), num_labeled))

        # Create masked labels
        self.masked_labels = []
        for idx, sample_idx in enumerate(self.samples_idx):
            if idx in labeled_indices:
                self.masked_labels.append(self.labels[sample_idx])
            else:
                self.masked_labels.append(-1)  # -1 indicates masked label

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        actual_idx = self.samples_idx[index]
        img, _ = self.dataset[actual_idx]
        label = self.masked_labels[index]

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        return (img1, img2), label

    def __len__(self) -> int:
        return len(self.samples_idx)


class CustomImageNetDataset(Dataset):
    '''
    Custom ImageNet and Tiny-ImageNet dataset with class-splitting and label percentage support for contrastive learning.
    '''

    def __init__(self, root, split='train', transform=None, dataset="imagenet",
                 class_percentages: Optional[Dict[int, float]] = None,
                 label_percentage: float = 1.0):
        """
        Args:
            root: Root directory of the dataset
            split: 'train' or 'val'
            transform: Data transformations
            dataset: "imagenet" or "tiny_imagenet"
            class_percentages: Dictionary mapping class indices to percentage of samples to keep (0.0 to 1.0)
                             e.g., {0: 0.5, 1: 1.0} keeps 50% of class 0 and 100% of class 1
            label_percentage: Percentage of samples that will retain their labels (0.0 to 1.0)
                            The rest will have their labels masked with -1
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.class_percentages = class_percentages
        self.label_percentage = label_percentage

        # Check if the dataset is already extracted
        if not os.path.exists(os.path.join(root, split)):
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    self._extract_dataset(dataset)
                dist.barrier()
            else:
                self._extract_dataset(dataset)

        if dataset.startswith("tiny"):
            self.dataset = TinyImageNet(root=root, split=split, download=True)
        else:
            self.dataset = datasets.ImageNet(root=root, split=split)

        self._setup_labels()

        if self.class_percentages is not None:
            self._apply_class_splitting()

        if self.label_percentage < 1.0:
            self._apply_label_masking()

    def _extract_dataset(self, dataset):
        # Trigger the extraction
        if dataset.startswith("tiny"):
            TinyImageNet(root=self.root, split=self.split, download=True)
        else:
            datasets.ImageNet(root=self.root, split=self.split)

    def _setup_labels(self):
        self.labels = self.dataset.targets
        self.samples_idx = list(range(len(self.dataset)))
        self.masked_labels = self.labels.copy()  # Create a copy for label masking

    def _apply_class_splitting(self):
        """Apply class-splitting based on specified percentages"""
        # Group indices by class
        class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Apply splitting and create final indices list
        final_indices = []
        for class_idx, indices in class_indices.items():
            percentage = self.class_percentages.get(class_idx, 1.0)  # Default to 100% if not specified
            num_samples = int(len(indices) * percentage)
            if num_samples > 0:
                selected_indices = random.sample(indices, num_samples)
                final_indices.extend(selected_indices)

        # Update samples index list
        self.samples_idx = sorted(final_indices)

    def _apply_label_masking(self):
        """Mask labels for a percentage of samples"""
        num_samples = len(self.samples_idx)
        num_labeled = int(num_samples * self.label_percentage)

        # Randomly select indices to keep labels
        labeled_indices = set(random.sample(range(num_samples), num_labeled))

        # Create masked labels
        self.masked_labels = []
        for idx, sample_idx in enumerate(self.samples_idx):
            if idx in labeled_indices:
                self.masked_labels.append(self.labels[sample_idx])
            else:
                self.masked_labels.append(-1)  # -1 indicates masked label

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        actual_idx = self.samples_idx[index]
        img, _ = self.dataset[actual_idx]
        label = self.masked_labels[index]

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        return (img1, img2), label

    def __len__(self) -> int:
        return len(self.samples_idx)


class CustomDataModule(pl.LightningDataModule):
    '''
    Custom datamodule for contrastive learning with class-splitting and label percentage support.
    '''
    def __init__(self, data_dir='data', batch_size=32, dataset="cifar100", num_workers=9,
                 loss="supcon", class_percentages: Optional[Dict[int, float]] = None,
                 label_percentage: float = 1.0):
        '''
        Args:
            data_dir: Base directory for dataset
            batch_size: Batch size for training
            dataset: Dataset name ('cifar100', 'cifar10', 'tiny_imagenet', 'imagenet')
            num_workers: Number of data loading workers
            loss: Loss function type ('ntx_ent' or 'supcon')
            class_percentages: Dictionary mapping class indices to percentage of samples to keep
            label_percentage: Percentage of samples that will retain their labels (0.0 to 1.0)
        '''
        super().__init__()
        self.data_dir = data_dir + "_" + dataset
        self.batch_size = batch_size
        self.dataset = dataset
        self.loss = loss
        self.class_percentages = class_percentages
        self.label_percentage = label_percentage

        if self.dataset.startswith("cifar"):
            resize_size = 32
            crop_size = 32
            if self.dataset == "cifar100":
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
            self.train_transform = transforms.Compose([
                transforms.RandAugment(num_ops=3, magnitude=18),
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
        if self.dataset == "cifar100":
            self.train_dataset = CustomCIFAR100Dataset(
                self.data_dir,
                train=True,
                transform=self.train_transform,
                class_percentages=self.class_percentages,
                label_percentage=self.label_percentage
            )
            self.val_dataset = CustomCIFAR100Dataset(
                self.data_dir,
                train=False,
                transform=self.val_transform,
                class_percentages=self.class_percentages,
                label_percentage=1.0  # Always use all labels for validation
            )
        elif self.dataset == "imagenet":
            self.train_dataset = CustomImageNetDataset(
                self.data_dir,
                split='train',
                transform=self.train_transform,
                dataset="imagenet",
                class_percentages=self.class_percentages,
                label_percentage=self.label_percentage
            )
            self.val_dataset = CustomImageNetDataset(
                self.data_dir,
                split='val',
                transform=self.val_transform,
                dataset="imagenet",
                class_percentages=self.class_percentages,
                label_percentage=1.0  # Always use all labels for validation
            )
        elif self.dataset == "tiny_imagenet":
            self.train_dataset = CustomImageNetDataset(
                self.data_dir,
                split='train',
                transform=self.train_transform,
                dataset="tiny_imagenet",
                class_percentages=self.class_percentages,
                label_percentage=self.label_percentage
            )
            self.val_dataset = CustomImageNetDataset(
                self.data_dir,
                split='val',
                transform=self.val_transform,
                dataset="tiny_imagenet",
                class_percentages=self.class_percentages,
                label_percentage=1.0  # Always use all labels for validation
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

class CustomEvaluationDataModule(pl.LightningDataModule):
    '''
    Custom datamodule for evaluation supporting:
    Classification: Food101, CIFAR10, CIFAR100, Birdsnap, SUN397, Cars, Aircraft, DTD
    Detection: VOC2007, OxfordPets, Caltech101, Flowers102
    '''

    def __init__(self, data_dir: str = './data', batch_size: int = 32,
                 dataset_type: Optional[str] = 'cifar100',
                 task: Literal['classification', 'detection'] = 'classification',
                 num_workers: int = 9):
        '''
        Parameters:
            data_dir (str): The base directory where the dataset is located
            batch_size (int): The number of data samples in each batch
            dataset_type (str): Type of dataset to use
            task (str): Whether to use classification or detection setup
            num_workers (int): The number of worker processes for data loading
        '''
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.task = task
        self.num_workers = num_workers

        # Validate dataset and task combination
        self.classification_datasets = {'food101', 'cifar10', 'cifar100', 'birdsnap',
                                        'sun397', 'stanfordcars', 'aircraft', 'dtd'}
        self.detection_datasets = {'voc2007', 'oxfordpets', 'caltech101', 'flowers102'}

        if task == 'classification' and dataset_type not in self.classification_datasets:
            raise ValueError(f"{dataset_type} is not a classification dataset")
        if task == 'detection' and dataset_type not in self.detection_datasets:
            raise ValueError(f"{dataset_type} is not a detection dataset")

        # Set up dataset-specific configurations
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup appropriate transforms based on dataset type"""
        # Set up basic parameters
        if self.dataset_type in ['cifar10', 'cifar100']:
            self.resize_size = 32
            self.crop_size = 32
            normalize = transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        else:  # All other datasets use ImageNet normalization
            self.resize_size = 256
            self.crop_size = 224
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

        # Set up transformations based on task
        if self.task == 'classification':
            if self.dataset_type in ['cifar10', 'cifar100']:
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
            else:
                self.train_transform = transforms.Compose([
                    transforms.Resize(self.resize_size),
                    transforms.RandomCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        else:  # detection task
            self.train_transform = transforms.Compose([
                transforms.Resize((self.crop_size, self.crop_size)),
                transforms.ToTensor(),
                normalize,
            ])

        # Val transform is always just resize and normalize
        self.val_transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
            normalize,
        ])

    def setup(self, stage=None):
        """Setup datasets based on type"""
        # Classification datasets
        if self.dataset_type == 'cifar10':
            self.train_dataset = datasets.CIFAR10(
                self.data_dir, train=True, transform=self.train_transform, download=True)
            self.val_dataset = datasets.CIFAR10(
                self.data_dir, train=False, transform=self.val_transform, download=True)

        elif self.dataset_type == 'cifar100':
            self.train_dataset = datasets.CIFAR100(
                self.data_dir, train=True, transform=self.train_transform, download=True)
            self.val_dataset = datasets.CIFAR100(
                self.data_dir, train=False, transform=self.val_transform, download=True)

        elif self.dataset_type == 'food101':
            self.train_dataset = datasets.Food101(
                self.data_dir, split='train', transform=self.train_transform, download=True)
            self.val_dataset = datasets.Food101(
                self.data_dir, split='test', transform=self.val_transform, download=True)

        elif self.dataset_type == 'birdsnap':
            self.train_dataset = datasets.Birdsnap(
                self.data_dir, split='train', transform=self.train_transform, download=True)
            self.val_dataset = datasets.Birdsnap(
                self.data_dir, split='test', transform=self.val_transform, download=True)

        elif self.dataset_type == 'sun397':
            self.train_dataset = datasets.SUN397(
                self.data_dir, transform=self.train_transform, download=True)
            self.val_dataset = datasets.SUN397(
                self.data_dir, transform=self.val_transform, download=True)

        elif self.dataset_type == 'stanfordcars':
            self.train_dataset = datasets.StanfordCars(
                self.data_dir, split='train', transform=self.train_transform, download=True)
            self.val_dataset = datasets.StanfordCars(
                self.data_dir, split='test', transform=self.val_transform, download=True)

        elif self.dataset_type == 'aircraft':
            self.train_dataset = datasets.FGVCAircraft(
                self.data_dir, split='trainval', transform=self.train_transform, download=True)
            self.val_dataset = datasets.FGVCAircraft(
                self.data_dir, split='test', transform=self.val_transform, download=True)

        elif self.dataset_type == 'dtd':
            self.train_dataset = datasets.DTD(
                self.data_dir, split='train', transform=self.train_transform, download=True)
            self.val_dataset = datasets.DTD(
                self.data_dir, split='test', transform=self.val_transform, download=True)

        # Detection datasets
        elif self.dataset_type == 'voc2007':
            self.train_dataset = datasets.VOCDetection(
                self.data_dir, year='2007', image_set='train',
                transform=self.train_transform, download=True)
            self.val_dataset = datasets.VOCDetection(
                self.data_dir, year='2007', image_set='val',
                transform=self.val_transform, download=True)

        elif self.dataset_type == 'oxfordpets':
            self.train_dataset = datasets.OxfordIIITPet(
                self.data_dir, split='trainval', transform=self.train_transform, download=True)
            self.val_dataset = datasets.OxfordIIITPet(
                self.data_dir, split='test', transform=self.val_transform, download=True)

        elif self.dataset_type == 'caltech101':
            dataset = datasets.Caltech101(self.data_dir, download=True)
            total_size = len(dataset)
            train_size = int(0.8 * total_size)
            indices = torch.randperm(total_size).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            self.train_dataset = torch.utils.data.Subset(
                datasets.Caltech101(self.data_dir, transform=self.train_transform), train_indices)
            self.val_dataset = torch.utils.data.Subset(
                datasets.Caltech101(self.data_dir, transform=self.val_transform), val_indices)

        elif self.dataset_type == 'flowers102':
            self.train_dataset = datasets.Flowers102(
                self.data_dir, split='train', transform=self.train_transform, download=True)
            self.val_dataset = datasets.Flowers102(
                self.data_dir, split='test', transform=self.val_transform, download=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
    