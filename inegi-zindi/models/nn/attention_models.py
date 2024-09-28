import torch
import torch.nn as nn

import torch
import torch.nn as nn

class FCAttention(nn.Module):
    """
    Fully Connected (FC) Attention mechanism as used in attention-based neural networks.

    This module applies attention to its input by using global information 
    to selectively emphasise informative features.

    Args:
        in_features (int): The number of input features the module should expect from its input.
        reduction_ratio (int, optional): The reduction ratio for the attention operation. 
            This controls the number of output features from the attention operation. 
            Default is 16.

    Attributes:
        avg_pool (nn.Module): Adaptive average pooling operation.
        fc (nn.Module): A sequence of linear and non-linear operations that implement the attention mechanism.
    """

    def __init__(self, in_features: int, reduction_ratio: int = 16):
        super(FCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction_ratio, in_features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FCAttention block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)

        Returns:
            torch.Tensor: Output tensor after applying the attention operation.
        """
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(2)).squeeze(2)
        y = self.fc(y).view(b, c)
        return x * y