import h5py
import numpy as np
import os

def optimize_hdf5_chunking(input_hdf5_file, output_hdf5_file, 
                           chunk_size=(32, 16, 16, 6), 
                           image_dtype=None, label_dtype=None):
    """
    Optimizes the chunk size of an HDF5 dataset and preserves metadata and dtype from the original file,
    unless a specific dtype is provided. The label dataset chunk size will use the first dimension of
    the chunk_size tuple.
    
    Parameters:
    -----------
    input_hdf5_file : str
        Path to the input HDF5 file containing datasets such as 'images' and 'labels'.
    
    output_hdf5_file : str
        Path to the output HDF5 file where the optimized dataset will be stored.
    
    chunk_size : tuple, optional
        The chunk size for the 'images' dataset in the new HDF5 file. The first dimension will be used
        for the 'labels' dataset. Default is (32, 16, 16, 6), suitable for batch processing of image data.
    
    image_dtype : numpy.dtype, optional
        Data type to be used for the 'images' dataset. If None, uses the original dtype from the input file.
    
    label_dtype : numpy.dtype, optional
        Data type to be used for the 'labels' dataset. If None, uses the original dtype from the input file.
    
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
            
            # Extract metadata for 'images'
            dtype_images = image_dtype if image_dtype is not None else hdf['images'].dtype
            band_names = hdf['images'].attrs.get('band_names', None)  # Get band names if available
            
            # Extract the images (with dtype conversion if specified)
            X = np.array(hdf['images'], dtype=dtype_images)
            
            # Extract metadata for 'labels'
            dtype_labels = label_dtype if label_dtype is not None else hdf['labels'].dtype
            
            # Extract the labels (with dtype conversion if specified)
            y = np.array(hdf['labels'], dtype=dtype_labels)
            
    except OSError as e:
        raise IOError(f"Error reading the HDF5 file: {e}")
    
    try:
        # Calculate the label chunk size using only the first dimension of the image chunk size
        label_chunk_size = (chunk_size[0],)
        
        # Create a new HDF5 file and apply optimized chunk size
        with h5py.File(output_hdf5_file, 'w') as new_hdf:
            # Create the 'images' dataset with optimized chunk size and specified dtype
            images_dset = new_hdf.create_dataset('images', data=X, 
                                                 chunks=chunk_size, 
                                                 compression='gzip', 
                                                 dtype=dtype_images)
            
            # Preserve metadata like 'band_names' in the new dataset
            if band_names is not None:
                images_dset.attrs['band_names'] = band_names
            
            # Create the 'labels' dataset with chunk size derived from the first dimension of chunk_size
            new_hdf.create_dataset('labels', data=y, 
                                   chunks=label_chunk_size, 
                                   compression='gzip', 
                                   dtype=dtype_labels)
        
        print(f"New HDF5 file created at {output_hdf5_file} with optimized chunking.")
    
    except OSError as e:
        raise IOError(f"Error writing the HDF5 file: {e}")


def balance_hdf5_dataset(hdf5_file, random_seed=None, dtype=None):
    """
    Balances a binary-labeled dataset stored in an HDF5 file by undersampling the majority class.
    
    Parameters:
    -----------
    hdf5_file : str
        Path to the input HDF5 file containing 'images' and 'labels' datasets.
        
    random_seed : int, optional
        Seed for reproducibility of the random sampling process. Default is None.
    
    dtype : numpy.dtype, optional
        Data type for reading the dataset. If None, uses the original dtype from the input file.
        
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
        # Extract the images (X) and labels (y) with the specified or default dtype
        if 'images' not in hdf or 'labels' not in hdf:
            raise ValueError("Datasets 'images' or 'labels' not found in the HDF5 file.")
        
        dtype_images = dtype if dtype is not None else hdf['images'].dtype
        X = np.array(hdf['images'], dtype=dtype_images)
        y = np.array(hdf['labels'], dtype=np.uint8)
        
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


def write_to_hdf5(X, y, output_hdf5_file, chunk_size=(32, 16, 16, 6), image_dtype=None, label_dtype=None):
    """
    Writes the input data (X, y) to an HDF5 file with specified chunk sizes and dtypes. The label dataset
    will use the first dimension of chunk_size as its chunk size.
    
    Parameters:
    -----------
    X : np.ndarray
        The input array of images.
        
    y : np.ndarray
        The input array of labels.
        
    output_hdf5_file : str
        Path to the output HDF5 file where the data will be stored.
        
    chunk_size : tuple, optional
        The chunk size for the 'images' dataset in the new HDF5 file. The label dataset will use only 
        the first dimension for its chunk size.
        Default is (32, 16, 16, 6), suitable for batch processing of image data.
    
    image_dtype : numpy.dtype, optional
        Data type for the 'images' dataset. If None, uses the dtype of the input array X.
        
    label_dtype : numpy.dtype, optional
        Data type for the 'labels' dataset. If None, uses the dtype of the input array y.
        
    Returns:
    --------
    None
        The function creates a new HDF5 file with the input data and specified chunk sizes and dtypes.
    """
    
    # Calculate the label chunk size using only the first dimension of the image chunk size
    label_chunk_size = (chunk_size[0],)
    
    # Create a new HDF5 file and write the input data
    with h5py.File(output_hdf5_file, 'w') as new_hdf:
        # Create the 'images' dataset with specified chunk size and dtype
        new_hdf.create_dataset('images', data=X, 
                               chunks=chunk_size, 
                               compression='gzip', 
                               dtype=image_dtype if image_dtype else X.dtype)
        
        # Create the 'labels' dataset with chunk size derived from the first dimension of chunk_size
        new_hdf.create_dataset('labels', data=y, 
                               chunks=label_chunk_size, 
                               compression='gzip', 
                               dtype=label_dtype if label_dtype else y.dtype)
    
    print(f"Data written to HDF5 file: {output_hdf5_file}")
