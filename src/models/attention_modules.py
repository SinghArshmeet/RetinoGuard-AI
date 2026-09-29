import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Woo et al., CBAM).
    Squeezes spatial dimensions via both Average and Max Pooling,
    passing features through a shared MLP to dynamically weight feature channels.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(in_planes // ratio, 8)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (Woo et al., CBAM).
    Compresses channel information via channel-wise Average and Max Pooling,
    followed by a 7x7 convolution to generate spatial attention maps for tumor localization.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention and Spatial Attention.
    """
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_planes, ratio=ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = x * self.channel_att(x)
        x_out = x_out * self.spatial_att(x_out)
        return x_out

class SqueezeAndExcitation(nn.Module):
    """Classic Squeeze-and-Excitation channel attention block."""
    def __init__(self, in_channels: int, reduction: int = 16):
        super(SqueezeAndExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
