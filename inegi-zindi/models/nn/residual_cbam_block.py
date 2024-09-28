import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam_block import CBAMBlock

# Residual block with CBAM attention
class ResidualBlockCBAM(nn.Module):
    """
    Residual Block with Convolutional Block Attention Module (CBAM) as described in the paper 
    "CBAM: Convolutional Block Attention Module" (https://arxiv.org/abs/1807.06521).

    This module applies both channel and spatial attention to its input and includes a residual connection.

    Args:
        in_channels (int): The number of input channels the module should expect from its input.
        out_channels (int): The number of output channels the module should produce.
        reduction (int, optional): The reduction ratio for the squeeze-and-excitation block. 
            This controls the number of output channels from the squeeze operation. 
            Default is 16.
        dropout_rate (float, optional): The dropout rate to be used after the final activation function.
            Default is 0.5.

    Attributes:
        conv1 (nn.Module): The first convolutional layer.
        bn1 (nn.Module): The first batch normalization layer.
        conv2 (nn.Module): The second convolutional layer.
        bn2 (nn.Module): The second batch normalization layer.
        cbam (nn.Module): The CBAM block that performs the channel and spatial attention.
        dropout (nn.Module): The dropout layer for regularization.
        shortcut (nn.Module): The shortcut connection.
    """
    def __init__(self, in_channels, out_channels, reduction=16, dropout_rate=0.5):
        super(ResidualBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAMBlock(out_channels, reduction)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass of the ResidualBlockCBAM.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: Output tensor after applying the ResidualBlockCBAM.
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(residual)
        out = F.relu(out)
        out = self.dropout(out)  # Regularization with dropout
        return out