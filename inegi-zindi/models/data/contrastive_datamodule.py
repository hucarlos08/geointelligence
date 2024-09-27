import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch
import random
import numpy as np
from torchvision import transforms
from .contrastive_dataset import ContrastiveDataset  # Assuming you have this dataset

class ContrastiveDataModule(pl.LightningDataModule):
    def __init__(self, train_file, test_file, batch_size=32, transform=None, dtype=np.uint16, num_workers=4, seed=42, split_ratio=(0.8, 0.2)):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.transform = transform

        self.dtype = dtype

        # Use the split ratio for train/val (since the test dataset is separate)
        self.split_ratio = split_ratio

    @classmethod
    def from_config(cls, config):
        """
        Create a DataModule from a configuration dictionary.

        Args:
            config (dict): A dictionary with configuration parameters.
        
        Returns:
            LandsatDataModule: A configured instance of the DataModule.
        """
        train_file = config['train_file']
        test_file = config.get('test_file', None)
        batch_size = config.get('batch_size', 32)
        num_workers = config.get('num_workers', 4)
        seed = config.get('seed', 42)
        split_ratio = config.get('split_ratio', (0.8, 0.2))
        dtype = config.get('dtype', np.uint16)

        # Handle transforms
        transform = None
        if 'transform' in config:
            transform_config = config['transform']
            transform = cls._create_transform(transform_config)

        return cls(
            train_file=train_file,
            test_file=test_file,
            batch_size=batch_size,
            transform=transform,
            dtype=dtype,
            num_workers=num_workers,
            seed=seed,
            split_ratio=split_ratio,
        )

    def setup(self, stage=None):
        """
        Setup method to split the dataset into train/val sets.
        Called on every GPU in distributed settings.
        
        Args:
            stage (str): Current stage (fit, validate, test, or None).
        """
        # Load the train dataset
        full_train_dataset = ContrastiveDataset(self.train_file, transform=self.transform, dtype=self.dtype)
        
        # Calculate sizes for train/val splits
        train_size = int(self.split_ratio[0] * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size

        # Perform the random split for train/val
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Load the test dataset separately
        if stage == 'test' or stage is None:
            self.test_dataset = ContrastiveDataset(self.test_file, transform=self.transform, dtype=self.dtype)

    def prepare_data(self):
        """
        Prepare the data for training. This method is called only once (not per-GPU).
        """
        # Set random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @staticmethod
    def _create_transform(transform_config):
        """
        Create a composition of transforms based on the configuration.

        Args:
            transform_config (dict): A dictionary of transform configurations.

        Returns:
            torchvision.transforms.Compose: A composition of transforms.
        """
        transform_list = []
        for transform_name, params in transform_config.items():
            if transform_name == 'RandomHorizontalFlip':
                transform_list.append(transforms.RandomHorizontalFlip(**params))
            elif transform_name == 'RandomVerticalFlip':
                transform_list.append(transforms.RandomVerticalFlip(**params))
            elif transform_name == 'RandomRotation':
                transform_list.append(transforms.RandomRotation(**params))
            elif transform_name == 'Normalize':
                transform_list.append(transforms.Normalize(**params))
            elif transform_name == 'Resize':
                transform_list.append(transforms.Resize(**params))
            elif transform_name == 'CenterCrop':
                transform_list.append(transforms.CenterCrop(**params))
            elif transform_name == 'RandomCrop':
                transform_list.append(transforms.RandomCrop(**params))
            elif transform_name == 'ColorJitter':
                transform_list.append(transforms.ColorJitter(**params))
            elif transform_name == 'ToTensor':
                transform_list.append(transforms.ToTensor())
            # Add more transforms as needed

        return transforms.Compose(transform_list)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
