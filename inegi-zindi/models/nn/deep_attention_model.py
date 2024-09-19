import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
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
        out = self.se(out)
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class ResAttentionConvNet(nn.Module):
    def __init__(self, input_channels=6, embedding_size=256, num_classes=1, dropout_rate=0.20):
        super(ResAttentionConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.res1 = ResidualBlock(32, 64)
        self.res2 = ResidualBlock(64, 128)
        self.res3 = ResidualBlock(128, 256)
        self.res4 = ResidualBlock(256, 512)
        
        self.attention = nn.MultiheadAttention(512, num_heads=4, batch_first=True)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, embedding_size)
        self.fc3 = nn.Linear(embedding_size, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)  # New dropout after initial conv
        
        x = self.res1(x)
        x = self.dropout(x)  # New dropout after res1
        x = self.res2(x)
        x = self.dropout(x)  # New dropout after res2
        x = self.res3(x)
        x = self.dropout(x)
        x = self.res4(x)
        
        # Apply attention
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)  # (batch, h*w, channels)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

    @classmethod
    def from_config(cls, config):
        input_channels = config.get('input_channels', 6)
        embedding_size = config.get('embedding_size', 256)
        num_classes = config.get('num_classes', 1)
        return cls(input_channels=input_channels, embedding_size=embedding_size, num_classes=num_classes)