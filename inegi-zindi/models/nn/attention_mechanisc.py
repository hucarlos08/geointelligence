import torch
import torch.nn as nn
from typing import Type

class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)

    This module applies channel and spatial attention to input feature maps.

    Args:
        channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for the channel attention mechanism.
        spatial_kernel_size (int): Kernel size for the spatial attention convolution.
        activation (Type[nn.Module]): Activation function to use.
        apply_residual (bool): Whether to apply a residual connection.
        channel_first (bool): Whether to apply channel attention before spatial attention.
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
        activation: Type[nn.Module] = nn.ReLU,
        apply_residual: bool = True,
        channel_first: bool = True
    ):
        super(CBAMBlock, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction_ratio, activation)
        self.spatial_attention = SpatialAttention(spatial_kernel_size, activation)
        self.apply_residual = apply_residual
        self.channel_first = channel_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CBAM block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output tensor after applying channel and spatial attention
        """
        residual = x
        
        if self.channel_first:
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
        else:
            x = self.spatial_attention(x)
            x = self.channel_attention(x)
        
        if self.apply_residual:
            x = x + residual
        
        return x

class ChannelAttention(nn.Module):
    """
    Channel Attention Module

    This module applies channel attention to input feature maps.

    Args:
        channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for the channel attention mechanism.
        activation (Type[nn.Module]): Activation function to use.
    """

    def __init__(self, channels: int, reduction_ratio: int, activation: Type[nn.Module]):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            activation(),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the channel attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output tensor after applying channel attention
        """
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = self.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module

    This module applies spatial attention to input feature maps.

    Args:
        kernel_size (int): Kernel size for the spatial attention convolution.
        activation (Type[nn.Module]): Activation function to use.
    """

    def __init__(self, kernel_size: int, activation: Type[nn.Module]):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.activation = activation()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spatial attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output tensor after applying spatial attention
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.activation(out)
        out = self.sigmoid(out)
        return x * out