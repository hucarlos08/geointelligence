import torch
import torch.nn as nn
from typing import Type

from .attention_mechanisc import CBAMBlock

class ResidualBlockCBAM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        reduction: int = 16,
        dropout_rate: float = 0.5,
        activation: Type[nn.Module] = nn.ReLU,
        normalization: Type[nn.Module] = nn.BatchNorm2d
    ):
        """
        Residual Block with Channel Attention Module (CBAM) implementation.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride value for the convolutional layers. Defaults to 1.
            expansion (int, optional): Expansion factor for the number of channels. Defaults to 1.
            reduction (int, optional): Reduction factor for the channel attention mechanism. Defaults to 16.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.5.
            activation (Type[nn.Module], optional): Activation function to be used. Defaults to nn.ReLU.
            normalization (Type[nn.Module], optional): Normalization layer to be used. Defaults to nn.BatchNorm2d.
        """
        super(ResidualBlockCBAM, self).__init__()
        self.expansion = expansion
        expanded_channels = out_channels * expansion

        # Pre-activation batch normalization for input channels
        self.bn1_input = normalization(in_channels)  # Match input channels
        self.conv1 = nn.Conv2d(in_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        
        # Batch normalization for expanded channels
        self.bn1_expanded = normalization(expanded_channels)  # Match expanded channels
        
        self.conv2 = nn.Conv2d(expanded_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = normalization(out_channels)
        self.cbam = CBAMBlock(out_channels, reduction)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation(inplace=True)

        # Shortcut connection to match input and output dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                normalization(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Residual Block with CBAM.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        # Pre-activation on the input channels
        out = self.bn1_input(x)
        out = self.activation(out)
        
        # Apply conv1 to expand channels
        out = self.conv1(out)
        
        # Apply normalization and activation after expansion
        out = self.bn1_expanded(out)
        out = self.activation(out)
        
        # Apply conv2 and CBAM
        out = self.conv2(out)
        out = self.cbam(out)

        # Add the shortcut (residual connection)
        out += self.shortcut(x)
        
        # Apply dropout
        out = self.dropout(out)
        
        return out
