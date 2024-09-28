import torch
import torch.nn as nn

from .se_block import SEBlock

class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) as described in the paper 
    "CBAM: Convolutional Block Attention Module" (https://arxiv.org/abs/1807.06521).

    This module applies both channel and spatial attention to its input.

    Args:
        channel (int): The number of input channels the module should expect from its input.
        reduction (int, optional): The reduction ratio for the squeeze-and-excitation block. 
            This controls the number of output channels from the squeeze operation. 
            Default is 16.

    Attributes:
        channel_attention (nn.Module): The squeeze-and-excitation block that performs the channel attention.
        spatial_attention (nn.Module): The convolutional block that performs the spatial attention.
    """

    def __init__(self, channel: int, reduction: int = 16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = SEBlock(channel, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),  # Normalization
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CBAM block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channel, height, width)

        Returns:
            torch.Tensor: Output tensor after applying channel and spatial attention.
        """
        # Channel Attention
        x = self.channel_attention(x)
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.spatial_attention(spatial_attn)
        
        return x * spatial_attn