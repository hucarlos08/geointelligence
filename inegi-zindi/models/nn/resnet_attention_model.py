import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block implementation.
    This block is used to model channel-wise dependencies and recalibrate feature maps.
    """

    def __init__(self, channel, reduction=16, dropout_rate=0.0):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
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
        out = x * y.expand_as(x)
        out = self.dropout(out)  # Apply dropout after SEBlock if needed
        return out


class ResidualBlock(nn.Module):
    """
    Residual block implementation.
    This block consists of two convolutional layers with batch normalization and a skip connection.
    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, dropout_rate=dropout_rate)
        
        # Shortcut connection for residual block
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        """
        Forward pass of the residual block.
        Args:
            x: Input feature map tensor.
        Returns:
            out: Output feature map tensor after applying the residual block.
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # SE block with optional dropout
        out += self.shortcut(residual)  # Add shortcut
        out = F.relu(out)
        out = self.dropout(out)  # Apply dropout after block if needed
        return out


class ResAttentionConvNet(nn.Module):
    """
    Residual Attention Convolutional Neural Network (CNN) implementation.
    This model consists of convolutional layers, residual blocks, attention layer, and fully connected layers.
    """

    def __init__(self, input_channels=6, embedding_size=256, num_classes=1, dropout_rate=0.20):
        super(ResAttentionConvNet, self).__init__()

        # Convolutional and residual layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Applying dropout within the blocks
        self.res1 = ResidualBlock(16, 32, dropout_rate=dropout_rate)
        self.res2 = ResidualBlock(32, 64, dropout_rate=dropout_rate)
        self.res3 = ResidualBlock(64, 128, dropout_rate=dropout_rate)
        #self.res4 = ResidualBlock(256, 512, dropout_rate=dropout_rate)

        # Attention layer
        self.attention = nn.MultiheadAttention(128, num_heads=4, batch_first=True)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        #self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(128, embedding_size)
        self.fc3 = nn.Linear(embedding_size, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
    def feature_extractor(self, x):
        """
        Function to extract features (embeddings) before the final classification layer.
        Args:
            x: Input tensor.
        Returns:
            embeddings: Output tensor containing the extracted features.
        """
        # Convolution and residual blocks with dropout handled inside
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        #x = self.res4(x)

        # Apply attention
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)  # (batch, h*w, channels)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1).view(b, c, h, w)

        # Global average pooling
        x = self.global_pool(x).view(b, -1)
        #x = F.relu(self.fc1(x))
        embeddings = self.fc2(x)
        return embeddings

    def forward(self, x, return_features=False):
        """
        Forward pass for classification.
        Args:
            x: Input tensor.
        Returns:
            x: Output tensor after classification.
        """
        embeddings = self.feature_extractor(x)
        x = self.dropout(embeddings)  # Dropout applied to embeddings before final layer
        x = self.fc3(x)
        if return_features:
            return x, embeddings
        return x

    @classmethod
    def from_config(cls, config):
        """
        Create an instance of the ResAttentionConvNet class from a configuration dictionary.
        Args:
            config: Dictionary containing the configuration parameters.
        Returns:
            An instance of the ResAttentionConvNet class.
        """
        input_channels = config.get('input_channels', 6)
        embedding_size = config.get('embedding_size', 256)
        num_classes = config.get('num_classes', 1)
        dropout_rate = config.get('dropout_rate', 0.5)
        return cls(input_channels=input_channels, embedding_size=embedding_size, num_classes=num_classes, dropout_rate=dropout_rate)