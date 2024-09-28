import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block as described in the paper 
    "Squeeze-and-Excitation Networks" (https://arxiv.org/abs/1709.01507).

    This module applies channel-wise attention to its input by using global information 
    to selectively emphasise informative channels.

    Args:
        channel (int): The number of input channels the module should expect from its input.
        reduction (int, optional): The reduction ratio for the squeeze operation. 
            This controls the number of output channels from the squeeze operation. 
            Default is 16.

    Attributes:
        avg_pool (nn.Module): Adaptive average pooling operation.
        fc (nn.Module): A sequence of linear and non-linear operations that implement the squeeze and excitation.
    """

    def __init__(self, channel: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SE block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channel, height, width)

        Returns:
            torch.Tensor: Output tensor after applying the squeeze and excitation operation.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)