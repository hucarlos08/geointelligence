import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlockCBAM(nn.Module):
    """
    SEBlockCBAM is a class that implements the Squeeze-and-Excitation (SE) block with Channel Attention Module (CBAM).

    Args:
        channel (int): The number of input channels.
        reduction (int, optional): The reduction ratio for the channel dimension. Default is 16.

    Attributes:
        avg_pool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        fc (nn.Sequential): Sequential module consisting of linear layers and activation functions.

    Methods:
        forward(x): Performs forward pass of the SEBlockCBAM module.

    """
    def __init__(self, channel, reduction=16):
        super(SEBlockCBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Performs forward pass of the SEBlockCBAM module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying the SEBlockCBAM module, of shape (batch_size, channels, height, width).

        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# CBAM block: channel and spatial attention
class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = SEBlockCBAM(channel, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),  # Normalization
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        x = self.channel_attention(x)
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.spatial_attention(spatial_attn)
        
        return x * spatial_attn

# Residual block with CBAM attention
class ResidualBlockCBAM(nn.Module):
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
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(residual)
        out = F.relu(out)
        out = self.dropout(out)  # Regularization with dropout
        return out

# Full ResNet with CBAM and regularization
class ResAttentionConvNetCBAM(nn.Module):
    """
    ResAttentionConvNetCBAM is a class that implements a convolutional neural network with residual blocks and attention mechanism using CBAM.

    Args:
        input_channels (int, optional): The number of input channels. Default is 6.
        embedding_size (int, optional): The size of the embedding layer. Default is 256.
        num_classes (int, optional): The number of output classes. Default is 1.
        dropout_rate (float, optional): The dropout rate. Default is 0.5.

    Attributes:
        conv1 (nn.Conv2d): Convolutional layer with 32 output channels and kernel size 3.
        bn1 (nn.BatchNorm2d): Batch normalization layer.
        res1 (ResidualBlockCBAM): Residual block with CBAM attention mechanism.
        res2 (ResidualBlockCBAM): Residual block with CBAM attention mechanism.
        res3 (ResidualBlockCBAM): Residual block with CBAM attention mechanism.
        res4 (ResidualBlockCBAM): Residual block with CBAM attention mechanism.
        global_pool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        fc1 (nn.Linear): Fully connected layer with 512 input features and 256 output features.
        fc2 (nn.Linear): Fully connected layer with 256 input features and embedding_size output features.
        fc3 (nn.Linear): Fully connected layer with embedding_size input features and num_classes output features.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        feature_extractor(x): Extracts features from the input tensor.
        forward(x, return_features): Performs forward pass of the ResAttentionConvNetCBAM module.

    """
    def __init__(self, input_channels=6, embedding_size=256, num_classes=1, dropout_rate=0.5):
        super(ResAttentionConvNetCBAM, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks with CBAM
        self.res1 = ResidualBlockCBAM(32, 64, reduction=16, dropout_rate=dropout_rate)
        self.res2 = ResidualBlockCBAM(64, 128, reduction=16, dropout_rate=dropout_rate)
        self.res3 = ResidualBlockCBAM(128, 256, reduction=16, dropout_rate=dropout_rate)
        self.res4 = ResidualBlockCBAM(256, 512, reduction=16, dropout_rate=dropout_rate)
        
        # Global pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers for feature extraction
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, embedding_size)

        # Fully connected layer for classification
        self.fc3 = nn.Linear(embedding_size, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)

    def feature_extractor(self, x):
        """
        Extracts features from the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Extracted features tensor of shape (batch_size, embedding_size).

        """
        x = F.relu(self.bn1(self.conv1(x)))  # Apply ReLU after batch normalization
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        x = self.global_pool(x)  # Global pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.relu(self.fc1(x))  # Fully connected layer followed by ReLU
        x = self.dropout(x)  # Apply dropout after ReLU
        x = F.relu(self.fc2(x))  # ReLU after second fully connected layer
        x = self.dropout(x)  # Apply dropout again

        return x

    def forward(self, x, return_features=False):
        """
        Performs forward pass of the ResAttentionConvNetCBAM module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            return_features (bool, optional): Whether to return the extracted features. Default is False.

        Returns:
            torch.Tensor: Output tensor after applying the ResAttentionConvNetCBAM module, of shape (batch_size, num_classes).
            torch.Tensor: Extracted features tensor of shape (batch_size, embedding_size), if return_features is True.

        """
        features = self.feature_extractor(x)
        x = self.dropout(features)  # Dropout before final layer
        x = self.fc3(x)  # No activation here because it's typically handled by the loss function (e.g., sigmoid for BCE)
        
        if return_features:
            return x, features
        return x

    @classmethod
    def from_config(cls, config):
        """
        Creates an instance of ResAttentionConvNetCBAM from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ResAttentionConvNetCBAM: An instance of ResAttentionConvNetCBAM.

        """
        input_channels = config.get('input_channels', 6)
        embedding_size = config.get('embedding_size', 256)
        num_classes = config.get('num_classes', 1)
        dropout_rate = config.get('dropout_rate', 0.5)
        return cls(input_channels=input_channels, embedding_size=embedding_size, num_classes=num_classes, dropout_rate=dropout_rate)

    @staticmethod
    def get_class_name() -> str:
        return 'RASENet'


