import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def replace_invalid_pixels_with_median(image, invalid_value=-9999, window_size=3):
    # image is expected to be a PyTorch tensor of shape (C, H, W)
    # Create a mask for invalid values
    mask_invalid = (image == invalid_value)

    # Replace invalid values with NaN to exclude them from median calculation
    image = image.clone()  # Create a copy to avoid in-place modification
    image[mask_invalid] = float('nan')

    # Pad the image to handle edge cases
    padded_image = F.pad(image.unsqueeze(0), pad=(window_size//2, window_size//2, window_size//2, window_size//2), mode='reflect')
    
    # For each pixel, compute the median of the surrounding window
    patches = F.unfold(padded_image, kernel_size=window_size)
    patches = patches.view(image.shape[0], window_size*window_size, image.shape[1], image.shape[2])

    # Compute the median of valid pixels in each patch
    median_filtered = torch.nanmedian(patches, dim=1).values

    # Replace invalid pixels with the median values
    image[mask_invalid] = median_filtered[mask_invalid]

    return image

def normalize_image(image):
    """Normalize each channel of the image to [0, 1] range."""
    # image is expected to be a numpy array of shape (C, H, W)
    min_vals = image.min(axis=(1, 2), keepdims=True)
    max_vals = image.max(axis=(1, 2), keepdims=True)
    return (image - min_vals) / (max_vals - min_vals)

class LandsatDataset(Dataset):
    def __init__(self, hdf5_file, transform=None, window_size=3):
        # Load HDF5 file
        with h5py.File(hdf5_file, 'r') as hdf:
            self.images = np.array(hdf['images'])
            self.labels = np.array(hdf['labels'])
        
        # Transpose the images to (C, H, W) format if they're not already
        if self.images.shape[-1] == 6:  # Assuming 6 channels
            self.images = np.transpose(self.images, (0, 3, 1, 2))

        self.window_size = window_size
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        label = self.labels[idx]

        # Convert to PyTorch tensor for processing
        image = torch.from_numpy(image)

        # Replace invalid pixels with the median of the surrounding window
        image = replace_invalid_pixels_with_median(image, window_size=self.window_size)

        # Convert back to numpy for normalization
        image = image.numpy()

        # Normalize the image to [0, 1] range
        image = normalize_image(image)

        # Convert to PyTorch tensor
        image = torch.from_numpy(image)

        # Apply transforms (if any)
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)