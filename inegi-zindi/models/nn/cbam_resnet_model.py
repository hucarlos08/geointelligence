import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .lsoftmax_module import LSoftmaxBinary

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

class FCAttention(nn.Module):
    def __init__(self, in_features, reduction_ratio=16):
        super(FCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction_ratio, in_features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(2)).squeeze(2)
        y = self.fc(y).view(b, c)
        return x * y


class FinalLayerAttention(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(FinalLayerAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, num_classes),
            nn.Sigmoid()
        )
        self.value = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        attention_weights = self.attention(x)
        values = self.value(x)
        return (attention_weights * values).sum(dim=1, keepdim=True)


class CBAMResNet(nn.Module):
    def __init__(self, input_channels=6, num_classes=1, initial_channels=32, num_blocks=4, 
                 channel_multiplier=2, dropout_rate=0.5, embedding_size=128, lsoftmax_margin=1):
        super(CBAMResNet, self).__init__()
        
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
        
        # Replace the final classification layer with the attention-based layer
        self.fc_final_attention = FinalLayerAttention(embedding_size, num_classes)
        
        # Add L-softmax after attention
        self.lsoftmax = LSoftmaxBinary(in_features=embedding_size, margin=lsoftmax_margin)

        self.dropout = nn.Dropout(dropout_rate)

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
        logits = self.fc_final_attention(x)
        
        # Apply L-Softmax on the extracted features (pass labels during training)
        # Check if labels are provided (i.e., during training)
        # if self.training and labels is not None:
        #     # Apply L-Softmax during training (with margin)
        #     logits = self.lsoftmax(x, labels)
        # else:
        #     # During inference, no margin is applied, so treat it like regular logits
        #     logits = self.lsoftmax(x)  # Pass without labels to skip the margin
        
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