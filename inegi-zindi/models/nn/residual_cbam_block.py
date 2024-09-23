import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Union

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
        normalization: Type[nn.Module] = nn.BatchNorm2d,
        use_preactivation: bool = True
    ):
        super(ResidualBlockCBAM, self).__init__()
        self.expansion = expansion
        self.use_preactivation = use_preactivation
        expanded_channels = out_channels * expansion

        self.conv1 = nn.Conv2d(in_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = normalization(expanded_channels)
        self.conv2 = nn.Conv2d(expanded_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = normalization(out_channels)
        self.cbam = CBAMBlock(out_channels, reduction)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                normalization(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_preactivation:
            return self._forward_preactivation(x)
        else:
            return self._forward_postactivation(x)

    def _forward_preactivation(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.cbam(out)
        out += self.shortcut(x)
        out = self.dropout(out)
        return out

    def _forward_postactivation(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        out += self.shortcut(residual)
        out = self.activation(out)
        out = self.dropout(out)
        return out