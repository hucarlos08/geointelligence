import torch

def replace_atypical_values(image, atypical_value=-9999, replacement_func=torch.nanmean):
    """
    Replace atypical values in each channel of an image with a specified function (default: mean).
    
    Parameters:
    -----------
    image : torch.Tensor
        Input image with shape (C, W, H), where C is the number of channels.
    atypical_value : float
        Value to be replaced (default: -9999).
    replacement_func : function
        Function to calculate replacement value (default: torch.nanmean).
    
    Returns:
    --------
    torch.Tensor
        Image with atypical values replaced.
    """
    # Create a mask where atypical values are found
    atypical_mask = (image == atypical_value)
    
    # Iterate over each channel
    for c in range(image.shape[0]):  # Channel is the first dimension
        channel = image[c, :, :]
        
        # Mask the channel by replacing atypical values with NaN
        masked_channel = torch.where(atypical_mask[c, :, :], torch.nan, channel)
        
        # Calculate the replacement value for atypical values
        replacement_value = replacement_func(masked_channel)
        
        # Replace atypical values in the channel
        image[c, :, :] = torch.where(atypical_mask[c, :, :], replacement_value, channel)
    
    return image

def calculate_channel_stats(images):
    """
    Calculate min, max, mean, and variance for each channel across all images.
    
    Parameters:
    -----------
    images : torch.Tensor
        Dataset of images with shape (N, C, W, H), where N is the total number of images.
    
    Returns:
    --------
    mins, maxs, means, variances : torch.Tensor
        Per-channel statistics, each with shape (C,).
    """
    # Flatten spatial dimensions (W, H) and batch (N), keeping channels separate
    flattened_images = images.view(images.shape[0], images.shape[1], -1)
    
    # Calculate statistics per channel
    mins = torch.min(flattened_images, dim=-1)[0]
    maxs = torch.max(flattened_images, dim=-1)[0]
    means = torch.mean(flattened_images, dim=-1)
    variances = torch.var(flattened_images, dim=-1)
    
    return mins, maxs, means, variances

def normalize_image(image, mins, maxs):
    """
    Normalize the pixel values of an image to the range [0, 1] using channel-wise min-max scaling.
    
    Parameters:
    -----------
    image : torch.Tensor
        Input image with shape (C, W, H).
    mins, maxs : torch.Tensor
        Per-channel min and max values, each with shape (C,).
    
    Returns:
    --------
    torch.Tensor
        Normalized image with values between 0 and 1 for each channel.
    """
    # Avoid division by zero by adding a small epsilon where min equals max
    eps = 1e-8
    normalized_image = (image - mins[:, None, None]) / (maxs[:, None, None] - mins[:, None, None] + eps)
    
    return normalized_image
