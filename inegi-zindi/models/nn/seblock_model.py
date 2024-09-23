import torch
import torch.nn as nn
from typing import Type, Union

class SEBlock(nn.Module):
    """
    Improved Squeeze-and-Excitation (SE) block implementation.
    This block models channel-wise dependencies and recalibrates feature maps.
    channel: Number of input channels.
    reduction: Reduction factor for the SE block.
    dropout_rate: Dropout rate for the SE block.
    activation: Activation function to use in the SE block.
    use_layer_norm: Flag to use layer normalization in the SE block.
    use_residual: Flag to use residual connection in the SE block.
    """

    def __init__(
        self,
        channel: int,
        reduction: int = 16,
        dropout_rate: float = 0.0,
        activation: Type[nn.Module] = nn.ReLU,
        use_layer_norm: bool = True,
        use_residual: bool = True
    ):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.use_residual = use_residual

        fc_layers = []
        if use_layer_norm:
            fc_layers.append(nn.LayerNorm(channel))
        
        fc_layers.extend([
            nn.Linear(channel, channel // reduction, bias=False),
            activation(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        ])

        self.fc = nn.Sequential(*fc_layers)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SE block.
        Args:
            x: Input feature map tensor.
        Returns:
            out: Output feature map tensor after recalibration.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        if self.use_residual:
            out = x * y + x  # Residual connection
        else:
            out = x * y
        
        out = self.dropout(out)
        return out