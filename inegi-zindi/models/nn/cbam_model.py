import math
import torch
import torch.nn as nn
from typing import Type

from .seblock_model import SEBlock
from .residual_cbam_block import ResidualBlockCBAM


class ResAttnConvNet(nn.Module):
    def __init__(
        self, 
        input_channels=6, 
        initial_channels=32,  # Starting dimension
        embedding_size=256,  # Desired embedding size
        depth=2,  # Number of times to halve the embedding size
        num_classes=1, 
        reduction: int = 16,
        dropout_rate=0.5,
        activation: Type[nn.Module] = nn.ReLU,
        normalization: Type[nn.Module] = nn.BatchNorm2d
    ):
        super(ResAttnConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, initial_channels, kernel_size=3, padding=1)
        self.bn1 = normalization(initial_channels)
        self.activation = activation(inplace=True)
        
        # Calculate the number of residual blocks based on initial_channels and embedding_size
        num_residual_blocks = int(math.log2(embedding_size // initial_channels))

        # Dynamically create residual blocks that double channels until embedding size is reached
        self.res_blocks = nn.ModuleList()
        in_channels = initial_channels
        out_channels = initial_channels
        for _ in range(num_residual_blocks):
            out_channels = min(in_channels * 2, embedding_size)  # Double channels but cap at embedding size
            self.res_blocks.append(
                ResidualBlockCBAM(in_channels, out_channels, reduction=reduction, 
                                  dropout_rate=dropout_rate, activation=activation, 
                                  normalization=normalization)
            )
            in_channels = out_channels  # Update in_channels for the next block

        # Global pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Dynamically create fully connected layers to reduce embedding size
        self.fc_layers = nn.ModuleList()
        current_size = embedding_size
        for _ in range(depth):
            next_size = current_size // 2
            self.fc_layers.append(nn.Linear(current_size, next_size))
            current_size = next_size

        # Final classification layer
        self.fc_final = nn.Linear(current_size, num_classes)

    def feature_extractor(self, x):
        """
        Extracts features from the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Extracted features tensor of shape (batch_size, embedding_size).
        """
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        
        # Apply fully connected layers
        x = features
        for fc in self.fc_layers:
            x = self.activation(fc(x))
            x = self.dropout(x)
        
        # Final classification layer
        x = self.fc_final(x) # (batch_size, num_classes), No activation function applied here as it is included in the loss function
        
        if return_features:
            return x, features
        return x

    @classmethod
    def from_config(cls, config: dict) -> 'ResAttnConvNet':
        return cls(
            input_channels=config.get('input_channels', 6),
            initial_channels=config.get('initial_channels', 32),
            embedding_size=config.get('embedding_size', 256),
            num_classes=config.get('num_classes', 1),
            reduction=config.get('reduction', 16),
            dropout_rate=config.get('dropout_rate', 0.5),
            depth=config.get('depth', 2)  # Depth parameter for fully connected layers
        )

    @staticmethod
    def get_class_name() -> str:
        return 'RACNet'
