import h5py
import numpy as np
import os

def optimize_hdf5_chunking(input_hdf5_file, output_hdf5_file, new_image_chunk_size=(32, 16, 16, 6), new_label_chunk_size=(32,)):
    """
    Optimizes the chunk size of an HDF5 dataset and preserves metadata and dtype from the original file.
    
    Parameters:
    -----------
    input_hdf5_file : str
        Path to the input HDF5 file containing datasets such as 'images' and 'labels'.
    
    output_hdf5_file : str
        Path to the output HDF5 file where the optimized dataset will be stored.
    
    new_image_chunk_size : tuple, optional
        The chunk size for the 'images' dataset in the new HDF5 file. 
        Default is (32, 16, 16, 6), suitable for batch processing of image data.
    
    new_label_chunk_size : tuple, optional
        The chunk size for the 'labels' dataset in the new HDF5 file.
        Default is (32,), suitable for batch-wise label access.
    
    Returns:
    --------
    None
        The function creates a new HDF5 file with optimized chunking, preserving the metadata and dtype 
        of the original datasets. Prints a success message upon completion.
    
    Raises:
    -------
    FileNotFoundError:
        If the input HDF5 file does not exist at the specified path.
    
    ValueError:
        If the required datasets ('images', 'labels') or their metadata are missing in the input file.
    
    IOError:
        For any general I/O errors that may occur during reading or writing of files.
    """
    
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
            y = np.array(hdf['labels'])
            
            # Extract metadata for 'images'
            dtype_images = hdf['images'].dtype
            band_names = hdf['images'].attrs.get('band_names', None)  # Get band names if available
            
    except OSError as e:
        raise IOError(f"Error reading the HDF5 file: {e}")
    
    try:
        # Create a new HDF5 file and apply optimized chunk size
        with h5py.File(output_hdf5_file, 'w') as new_hdf:
            # Create the 'images' dataset with optimized chunk size and original dtype
            images_dset = new_hdf.create_dataset('images', data=X, 
                                                 chunks=new_image_chunk_size, 
                                                 compression='gzip', 
                                                 dtype=dtype_images)
            
            # Preserve metadata like 'band_names' in the new dataset
            if band_names is not None:
                images_dset.attrs['band_names'] = band_names
            
            # Create the 'labels' dataset with optimized chunk size and preserve dtype
            new_hdf.create_dataset('labels', data=y, 
                                   chunks=new_label_chunk_size, 
                                   compression='gzip', 
                                   dtype=y.dtype)
        
        print(f"New HDF5 file created at {output_hdf5_file} with optimized chunking.")
    
    except OSError as e:
        raise IOError(f"Error writing the HDF5 file: {e}")



def balance_hdf5_dataset(hdf5_file, random_seed=None):
    """
    Balances a binary-labeled dataset stored in an HDF5 file by undersampling the majority class.
    
    Parameters:
    -----------
    hdf5_file : str
        Path to the input HDF5 file containing 'images' and 'labels' datasets.
        
    random_seed : int, optional
        Seed for reproducibility of the random sampling process. Default is None.
        
    Returns:
    --------
    X_balanced : np.ndarray
        The balanced array of images.
        
    y_balanced : np.ndarray
        The balanced array of labels.
    """
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Open the HDF5 file
    with h5py.File(hdf5_file, 'r') as hdf:
        # Extract the images (X) and labels (y)
        if 'images' not in hdf or 'labels' not in hdf:
            raise ValueError("Datasets 'images' or 'labels' not found in the HDF5 file.")
        
        X = np.array(hdf['images'])
        y = np.array(hdf['labels'])
        
        # Check the shape of the data
        print("Shape of X (images):", X.shape)
        print("Shape of y (labels):", y.shape)
        
        # Count the number of 1's
        num_ones = np.sum(y == 1)
        
        # Get the indices of 1's and 0's
        ones_indices = np.where(y == 1)[0]
        zeros_indices = np.where(y == 0)[0]
        
        # Sample the same number of 0's as there are 1's
        balanced_zero_indices = np.random.choice(zeros_indices, num_ones, replace=False)
        
        # Combine the indices
        balanced_indices = np.concatenate([ones_indices, balanced_zero_indices])
        
        # Create balanced X and y
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        # Print the results
        print(f"Number of 1's in balanced y: {np.sum(y_balanced == 1)}")
        print(f"Number of 0's in balanced y: {np.sum(y_balanced == 0)}")
    
    return X_balanced, y_balanced
