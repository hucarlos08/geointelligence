import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class ContrastiveDataset(Dataset):
    def __init__(self, hdf5_file, transform=None, dtype=np.uint16):
        """
        Dataset class for binary classification with contrastive loss.
        
        Args:
            hdf5_file (str): Path to the HDF5 file containing images and labels.
            transform (callable, optional): A function/transform to apply to the images.
            dtype (np.dtype, optional): Data type of the images. Defaults to np.uint16.
        """
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.dtype = dtype
        
        # Open the HDF5 file once and keep it open
        self.hdf = h5py.File(self.hdf5_file, 'r')
        self.images = self.hdf['images']
        self.labels = np.array(self.hdf['labels'])  # Convert labels to numpy array for easy access
        self.dataset_length = len(self.images)

        # Group indices by class (binary: 0 or 1)
        self.class_to_indices = {0: [], 1: []}
        for idx, label in enumerate(self.labels):
            label = int(label)
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        Get a pair of samples and a label for contrastive loss.
        
        Args:
            idx (int): Index of the first sample (anchor).
        
        Returns:
            tuple: (anchor_image, pair_image, label)
        """
        # Select the anchor image and its label
        anchor_image = np.array(self.images[idx], dtype=self.dtype)
        anchor_label = int(self.labels[idx])

        # Decide randomly whether to return a positive or negative pair
        if random.random() > 0.5:
            # Positive pair: Same class as anchor
            pair_idx = random.choice(self.class_to_indices[anchor_label])
            label = 0  # Positive pair
        else:
            # Negative pair: Different class than anchor
            negative_label = 1 - anchor_label  # Flip between 0 and 1
            pair_idx = random.choice(self.class_to_indices[negative_label])
            label = 1  # Negative pair

        pair_image = np.array(self.images[pair_idx], dtype=self.dtype)
        pair_label = torch.tensor([self.labels[pair_idx]], dtype=torch.float32)

        # Normalize both images by channel
        anchor_image, pair_image = map(self.normalize_image, [anchor_image, pair_image])

        # Convert images to PyTorch tensors and permute to (6, 16, 16) format
        anchor_image = torch.from_numpy(np.transpose(anchor_image, (2, 0, 1)))
        pair_image = torch.from_numpy(np.transpose(pair_image, (2, 0, 1)))

        # Apply transformations, if any
        if self.transform:
            anchor_image = self.transform(anchor_image)
            pair_image = self.transform(pair_image)

        # Ensure the label has the shape [1]
        anchor_label = torch.tensor([anchor_label], dtype=torch.float32)

        return anchor_image, anchor_label, pair_image, pair_label, torch.tensor([label], dtype=torch.float32)

    def normalize_image(self, image):
        """
        Normalize the image on a per-channel basis.
        
        Args:
            image (np.ndarray): The image to be normalized.
        
        Returns:
            np.ndarray: The normalized image.
        """
        image = image.astype(np.float32)
        mins = np.min(image, axis=(0, 1))
        maxs = np.max(image, axis=(0, 1))
        return (image - mins) / (maxs - mins)

    def __del__(self):
        # Ensure the HDF5 file is closed when the dataset object is deleted
        self.hdf.close()
