import jax.numpy as jnp
from jax import vmap

def replace_atypical_values(image, atypical_value=-9999, replacement_func=jnp.nanmean):
    """
    Replace atypical values in each channel of a 16x16xC image with a specified function (default: mean).
    
    Parameters:
    -----------
    image : jnp.ndarray
        Input image with shape (16, 16, C), where C is the number of channels.
    atypical_value : float
        Value to be replaced (default: -9999).
    replacement_func : function
        Function to calculate replacement value (default: jnp.nanmean).
    
    Returns:
    --------
    jnp.ndarray
        Image with atypical values replaced.
    """
    def replace_channel(channel):
        replacement = replacement_func(jnp.where(channel == atypical_value, jnp.nan, channel))
        return jnp.where(channel == atypical_value, replacement, channel)
    
    return vmap(replace_channel, in_axes=2, out_axes=2)(image)

def calculate_channel_stats(images):
    """
    Calculate min, max, mean, and variance for each channel across all images.
    
    Parameters:
    -----------
    images : jnp.ndarray
        Dataset of images with shape (N, 16, 16, C), where N is the total number of images.
    
    Returns:
    --------
    mins, maxs, means, variances : jnp.ndarray
        Per-channel statistics, each with shape (C,).
    """
    mins = jnp.min(images, axis=(0, 1, 2))
    maxs = jnp.max(images, axis=(0, 1, 2))
    means = jnp.mean(images, axis=(0, 1, 2))
    variances = jnp.var(images, axis=(0, 1, 2))
    return mins, maxs, means, variances

def normalize_image(image, mins, maxs):
    """
    Normalize the pixel values of an image to the range [0, 1] using channel-wise min-max scaling.
    
    Parameters:
    -----------
    image : jnp.ndarray
        Input image with shape (16, 16, C).
    mins, maxs : jnp.ndarray
        Per-channel min and max values, each with shape (C,).
    
    Returns:
    --------
    jnp.ndarray
        Normalized image with values between 0 and 1 for each channel.
    """
    return (image - mins) / (maxs - mins)

def process_images(images, atypical_value=-9999):
    """
    Process the entire dataset of images by replacing atypical values, normalizing, and calculating statistics.
    
    Parameters:
    -----------
    images : jnp.ndarray
        Dataset of images with shape (N, 16, 16, C), where N is the total number of images.
    atypical_value : float
        Value to be replaced (default: -9999).
    
    Returns:
    --------
    images_normalized : jnp.ndarray
        Processed and normalized images.
    means, variances : jnp.ndarray
        Per-channel means and variances of the normalized images.
    """
    # Step 1: Replace atypical values
    images_cleaned = vmap(replace_atypical_values, in_axes=(0, None))(images, atypical_value)
    
    # Step 2: Calculate channel-wise statistics
    mins, maxs, _, _ = calculate_channel_stats(images_cleaned)
    
    # Step 3: Normalize images to [0, 1] channel-wise
    images_normalized = vmap(normalize_image, in_axes=(0, None, None))(images_cleaned, mins, maxs)
    
    # Step 4: Calculate statistics on normalized images
    _, _, means, variances = calculate_channel_stats(images_normalized)
    
    return images_normalized, means, variances

import json
import numpy as np

def save_stats(means, variances, filename='landsat_stats.json'):
    """
    Save mean and variance values to a JSON file.
    
    Parameters:
    -----------
    means : np.ndarray or list
        Array of mean values for each channel.
    variances : np.ndarray or list
        Array of variance values for each channel.
    filename : str, optional
        Name of the file to save the statistics (default is 'landsat_stats.json').
    """
    stats = {
        'means': means.tolist() if isinstance(means, np.ndarray) else means,
        'variances': variances.tolist() if isinstance(variances, np.ndarray) else variances
    }
    
    with open(filename, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Statistics saved to {filename}")

def load_stats(filename='landsat_stats.json'):
    """
    Load mean and variance values from a JSON file.
    
    Parameters:
    -----------
    filename : str, optional
        Name of the file to load the statistics from (default is 'landsat_stats.json').
    
    Returns:
    --------
    means : np.ndarray
        Array of mean values for each channel.
    variances : np.ndarray
        Array of variance values for each channel.
    """
    with open(filename, 'r') as f:
        stats = json.load(f)
    
    means = np.array(stats['means'])
    variances = np.array(stats['variances'])
    
    print(f"Statistics loaded from {filename}")
    return means, variances


# Example usage
# Assuming 'images' is a JAX array of shape (N, 16, 16, C) where N is the total number of images
# images_normalized, means, variances = process_images(images)