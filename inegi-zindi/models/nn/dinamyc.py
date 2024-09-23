# Importing the required libraries for PyTorch model creation
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-implementing the dynamic residual network with the necessary imports in place

class ResidualBlockCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, dropout_rate=0.0):
        super(ResidualBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAMBlock(out_channels, reduction)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        if self.dropout:
            out = self.dropout(out)
        out += self.shortcut(x)
        return F.relu(out)

class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = SEBlockCBAM(channel, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_out = self.channel_attention(x)
        max_pool = torch.max(x_out, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_out, dim=1, keepdim=True)
        attention = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.spatial_attention(attention)
        return x_out * attention

class SEBlockCBAM(nn.Module):
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
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DynamicResidualNetwork(nn.Module):
    def __init__(self, input_channels, num_blocks=4, initial_channels=32, channel_factor=2, 
                 reduction=16, dropout_rate=0.0, num_classes=10):
        super(DynamicResidualNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, initial_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(initial_channels)

        # Dynamically create the residual blocks
        channels = initial_channels
        self.residual_blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            next_channels = channels * channel_factor
            self.residual_blocks.append(
                ResidualBlockCBAM(channels, next_channels, reduction, dropout_rate)
            )
            channels = next_channels

        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for block in self.residual_blocks:
            out = block(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# This new dynamic residual network can now be initialized with varying configurations.

