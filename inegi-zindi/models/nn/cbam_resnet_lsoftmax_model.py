import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .lsoftmax_module import LSoftmaxBinary
from .attention_models import FCAttention
from .residual_cbam_block import ResidualBlockCBAM
from .cbam_resnet_model import CBAMResNet

class AttentionWeightedFeatures(nn.Module):
    """
    AttentionWeightedFeatures applies an attention mechanism to weight input features.
    
    Args:
        embedding_size (int): The number of input features.
    
    Attributes:
        attention (nn.Module): Computes attention weights for the features.
    """

    def __init__(self, embedding_size: int):
        super(AttentionWeightedFeatures, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, embedding_size),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionWeightedFeatures.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, embedding_size). 
            Each feature is weighted by its corresponding attention score.
        """
        attention_weights = self.attention(x)
        return x * attention_weights


class CBAMResNetLSoftmax(CBAMResNet):
    """
    CBAMResNetLSoftmax: A version of CBAMResNet that uses L-Softmax for classification.
    This class inherits from CBAMResNet and modifies the final layer to use L-Softmax.
    """

    def __init__(self, input_channels=6, num_classes=1, initial_channels=32, num_blocks=4, 
                 channel_multiplier=2, dropout_rate=0.5, embedding_size=128, lsoftmax_margin=1):
        """
        Initialize the CBAMResNetLSoftmax model.

        Args:
            input_channels (int): Number of input channels in the data.
            num_classes (int): Number of output classes (default is 1 for binary classification).
            initial_channels (int): Number of channels in the first convolutional layer.
            num_blocks (int): Number of residual blocks in the network.
            channel_multiplier (int): Factor by which the number of channels increases in each block.
            dropout_rate (float): Dropout rate for regularization.
            embedding_size (int): Size of the embedding before the final classification layer.
            lsoftmax_margin (int): Margin parameter for L-Softmax.
        """
        self.lsoftmax_margin = lsoftmax_margin
        super(CBAMResNetLSoftmax, self).__init__(input_channels, num_classes, initial_channels, 
                                                 num_blocks, channel_multiplier, dropout_rate, 
                                                 embedding_size)

    def _set_final_layer(self):
        self.weighted_features = AttentionWeightedFeatures(embedding_size=self.embedding_size)
        self.final_layer_classification = LSoftmaxBinary(in_features=self.embedding_size, margin=self.lsoftmax_margin)

        
    def forward(self, x, labels=None, return_features=False):
        features = self.feature_extractor(x)
        x = self.dropout(features)
        
        # Get weighted features
        weighted_features = self.weighted_features(x)
        
        # Apply L-Softmax on the weighted features
        if self.training and labels is not None:
            logits = self.final_layer_classification(weighted_features, labels)
        else:
            logits = self.final_layer_classification(weighted_features)
        
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
            embedding_size=config.get('embedding_size', 128),
            lsoftmax_margin=config.get('lsoftmax_margin', 1)
        )

    @staticmethod
    def get_class_name() -> str:
        return 'CBAMResNet-LSoftmax'