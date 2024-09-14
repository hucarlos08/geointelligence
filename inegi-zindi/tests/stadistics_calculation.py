import h5py
import numpy as np
import jax.numpy as jnp
import os

from models.data import process_images, save_stats

if __name__ == '__main__':

    input_hdf5_file = '/teamspace/studios/this_studio/dataset/optimized_train_data.h5'

    # Check if the input file exists
    if not os.path.exists(input_hdf5_file):
        raise FileNotFoundError(f"Input HDF5 file not found: {input_hdf5_file}")
    
    try:
        # Open the original HDF5 file and extract data, dtype, and metadata
        with h5py.File(input_hdf5_file, 'r') as hdf:
            # Check if required datasets exist in the input HDF5 file
            if 'images' not in hdf or 'labels' not in hdf:
                raise ValueError("Required datasets 'images' and/or 'labels' not found in the HDF5 file.")
            
            # Extract data
            X = np.array(hdf['images'])
        

        images_normalized, means, variances = process_images(X, atypical_value=-9999)
        print(f"Processed images with shape {images_normalized.shape} and statistics:")
        print(f"Means: {means}")
        print(f"Variances: {variances}")

        save_stats(means, variances, filename='landsat_stats.json')

    except OSError as e:
        raise IOError(f"Error reading the HDF5 file: {e}")
