import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class LandsatDataset(Dataset):
    def __init__(self, hdf5_file, transform=None, dtype=np.uint16):

        # Store the file path
        self.hdf5_file = hdf5_file
        # Store the transform
        self.transform = transform
        # Store the data type
        self.dtype = dtype
        
        # Open the HDF5 file once and keep it open
        self.hdf = h5py.File(self.hdf5_file, 'r')
        self.dataset_length = len(self.hdf['images'])

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # Read the data without reopening the file
        image = np.array(self.hdf['images'][idx], dtype=self.dtype)  # Shape: (16, 16, 6) unsigned int

        # Normalize the image by channel
        image = image.astype(np.float32)
        mins = np.min(image, axis=(0, 1))
        maxs = np.max(image, axis=(0, 1))
        image = (image - mins) / (maxs - mins)
        
        # Permute the image to shape (6, 16, 16)
        image = np.transpose(image, (2, 0, 1))  # Now the shape will be (6, 16, 16)
        
        label = self.hdf['labels'][idx]

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
        
