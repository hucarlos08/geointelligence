import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class LandsatDataset(Dataset):
    def __init__(self, hdf5_file, transform=None, invalid_value=-9999, normalize_factor=20000.0):
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.invalid_value = invalid_value
        self.normalize_factor = normalize_factor
        
        # Open the HDF5 file once and keep it open
        self.hdf = h5py.File(self.hdf5_file, 'r')
        self.dataset_length = len(self.hdf['images'])

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # Read the data without reopening the file
        image = self.hdf['images'][idx].astype(np.float32)  # Shape: (16, 16, 6)
        
        # Permute the image to shape (6, 16, 16)
        image = np.transpose(image, (2, 0, 1))  # Now the shape will be (6, 16, 16)
        
        label = self.hdf['labels'][idx]

        # Replace invalid pixels with 0
        image[image == self.invalid_value] = 0
        
        # Normalize the image
        image = image / self.normalize_factor
        
        # Convert image to PyTorch tensor
        image = torch.from_numpy(image)
        
        # Ensure the label has the shape [1]
        label = torch.tensor([label], dtype=torch.float32)
        
        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def __del__(self):
        # Ensure the HDF5 file is closed when the dataset object is deleted
        self.hdf.close()
