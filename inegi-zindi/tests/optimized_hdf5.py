import h5py
import numpy as np
import os

from models.data import optimize_hdf5_chunking


if __name__ == '__main__':

    try:
        input_hdf5_file = '/teamspace/studios/this_studio/datasets/train_data.h5'
        output_hdf5_file = '/teamspace/studios/this_studio/datasets/uint16_optimized_train_data.h5'
        chunk_size = (32, 16, 16, 6)
        image_dtype = np.uint16
        label_dtype = np.uint8
        optimize_hdf5_chunking(input_hdf5_file, output_hdf5_file, chunk_size=chunk_size, image_dtype=image_dtype, label_dtype=label_dtype)

    except OSError as e:
        raise IOError(f"Error reading the HDF5 file: {e}")
    finally:
        print("Optimized HDF5 chunking completed.")
    