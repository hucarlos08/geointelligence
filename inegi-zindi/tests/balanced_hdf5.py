import h5py
import numpy as np
import os

from models.data import balance_hdf5_dataset
from models.data import write_to_hdf5

if __name__ == '__main__':

    try:
        input_hdf5_file = '/teamspace/studios/this_studio/datasets/uint16_optimized_train_data.h5'
        X_balanced, y_balanced = balance_hdf5_dataset(input_hdf5_file, random_seed=42, dtype=np.uint16)
        print(f"Balanced images with shape {X_balanced.shape} and labels with shape {y_balanced.shape}")
        
        output_hdf5_file = '/teamspace/studios/this_studio/datasets/uint16_optimized_balanced_train_data.h5'
        chunk_size=(32, 16, 16, 6)
        image_dtype=np.uint16
        label_dtype=np.uint8
        write_to_hdf5(X_balanced, y_balanced, output_hdf5_file, chunk_size=chunk_size, image_dtype=image_dtype, label_dtype=label_dtype)

    except OSError as e:
        raise IOError(f"Error reading the HDF5 file: {e}")
    finally:
        print("Balanced HDF5 dataset completed.")