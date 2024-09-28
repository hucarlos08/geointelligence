import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .attention_models import FCAttention
from .residual_cbam_block import ResidualBlockCBAM

class AttentionWeightedClassifier(nn.Module):
    """
    AttentionWeightedClassifier is a neural network module that applies an attention mechanism 
    before performing classification. The attention mechanism allows the model to focus on 
    different parts of the input when making the final prediction.

    Args:
        embedding_size (int): The number of input features the module should expect from its input.
        num_classes (int): The number of output classes the model should predict.

    Attributes:
        attention (nn.Module): A sequence of linear and non-linear operations that implement the attention mechanism.
        value (nn.Module): A linear layer that transforms the input features into output classes.
    """

    def __init__(self, embedding_size: int, num_classes: int):
        super(AttentionWeightedClassifier, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, num_classes),
            nn.Sigmoid()
        )
        self.value = nn.Linear(embedding_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionWeightedClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1). Each element in the output tensor 
            represents the weighted sum of the class scores for a single instance in the batch. 
            The weights are determined by the attention mechanism.
        """
        attention_weights = self.attention(x)
        values = self.value(x)
        return (attention_weights * values).sum(dim=1, keepdim=True)


class CBAMResNet(nn.Module):
    def __init__(self, input_channels=6, num_classes=1, initial_channels=32, num_blocks=4, 
                 channel_multiplier=2, dropout_rate=0.5, embedding_size=128):
        super(CBAMResNet, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.initial_channels = initial_channels
        self.num_blocks = num_blocks
        self.channel_multiplier = channel_multiplier
        self.dropout_rate = dropout_rate
        self.embedding_size = embedding_size
        
        self.conv1 = nn.Conv2d(input_channels, initial_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        
        # Dynamic creation of residual blocks
        self.res_blocks = nn.ModuleList()
        in_channels = initial_channels
        for i in range(num_blocks):
            out_channels = in_channels * channel_multiplier
            self.res_blocks.append(ResidualBlockCBAM(in_channels, out_channels, reduction=16, dropout_rate=dropout_rate))
            in_channels = out_channels
        
        # Global pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
         # Modify the fc_layers to include attention
        self.fc_layers = nn.ModuleList()
        self.fc_attention_layers = nn.ModuleList()
        fc_in_features = in_channels
        while fc_in_features > embedding_size:
            fc_out_features = fc_in_features // 2
            self.fc_layers.append(nn.Linear(fc_in_features, fc_out_features))
            self.fc_attention_layers.append(FCAttention(fc_out_features))
            fc_in_features = fc_out_features

        self.dropout = nn.Dropout(dropout_rate)

        self._set_final_layer()
    
        
    def _set_final_layer(self):
        self.final_layer_classification = AttentionWeightedClassifier(self.embedding_size, self.num_classes)

    def feature_extractor(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        for fc_layer, attention_layer in zip(self.fc_layers, self.fc_attention_layers):
            x = F.relu(fc_layer(x))
            x = attention_layer(x)
            x = self.dropout(x)
        
        return x
        
    def forward(self, x, labels=None, return_features=False):
        features = self.feature_extractor(x)
        x = self.dropout(features)
        
        # Apply FinalLayerAttention first
        logits = self.final_layer_classification(x)

        if return_features:
            return logits, features
        return logits

    @classmethod
    def from_config(cls, config):
        return cls(
            input_channels=config.get('input_channels', 6),
            num_classes=config.get('num_classes', 1),
            initial_channels=config.get('initial_channels', 32),
            num_blocks=config.get('num_blocks', 4),
            channel_multiplier=config.get('channel_multiplier', 2),
            dropout_rate=config.get('dropout_rate', 0.5),
            embedding_size=config.get('embedding_size', 128)
        )

    @staticmethod
    def get_class_name() -> str:
        return 'CBAMResNet'