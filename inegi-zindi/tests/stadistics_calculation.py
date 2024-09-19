import h5py
import numpy as np
import jax.numpy as jnp
import os

from models.data import process_images, save_stats

if __name__ == '__main__':

    PATH = '/teamspace/studios/this_studio/datasets/'
    HDF5_FILE_NAME = 'test_data.h5'

    input_hdf5_file = os.path.join(PATH, HDF5_FILE_NAME)

    # Check if the input file exists
    if not os.path.exists(input_hdf5_file):
        raise FileNotFoundError(f"Input HDF5 file not found: {input_hdf5_file}")
    
    try:
        # Open the original HDF5 file and extract data, dtype, and metadata
        with h5py.File(input_hdf5_file, 'r') as hdf:
            # Check if required datasets exist in the input HDF5 file
            if 'images' not in hdf:
                raise ValueError("Required datasets 'images' and/or 'labels' not found in the HDF5 file.")
            
            # Lee el dataset como int16
            X = hdf['images'][:].astype(np.int16)

        
        print(f"Loaded images with shape {X.shape}")

        images_normalized, means, variances = process_images(X, atypical_value=-9999)
        print(f"Processed images with shape {images_normalized.shape} and statistics:")
        print(f"Means: {means}")
        print(f"Variances: {variances}")

        # Save the calculated statistics to a JSON file
        means = np.array(means)
        variances = np.array(variances)

        stats_file_name = 'landsat_stats_uint.json'
        stats_file = os.path.join(PATH, stats_file_name)
        save_stats(means, variances, filename=stats_file)
        print(f"Statistics saved to {stats_file}")

    except OSError as e:
        raise IOError(f"Error reading the HDF5 file: {e}")
