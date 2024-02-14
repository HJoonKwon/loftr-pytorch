import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_ch, out_ch, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_ch, out_ch, stride=stride), nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.block = nn.Sequential(
            ResNetBlock(in_ch, out_ch, stride=stride),
            ResNetBlock(out_ch, out_ch, stride=1),
        )

    def forward(self, x):
        return self.block(x)


class FPNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.skip = conv1x1(out_ch, in_ch)
        self.merge = nn.Sequential(
            conv3x3(in_ch, in_ch),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(),
            conv3x3(in_ch, out_ch),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        skip = self.skip(skip)
        x = self.merge(x + skip)
        return x


class ResNetFPN_8_2(nn.Module):
    def __init__(self, config):
        super().__init__()
        initial_dim = config["initial_dim"]
        block_dims = config["block_dims"]

        # downsample(1/2)
        self.enc0 = nn.Sequential(
            nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(initial_dim),
            nn.ReLU(inplace=True),
        )
        self.enc1 = EncoderBlock(initial_dim, block_dims[0], stride=1)  # 1/2
        self.enc2 = EncoderBlock(block_dims[0], block_dims[1], stride=2)  # 1/4
        self.enc3 = EncoderBlock(block_dims[1], block_dims[2], stride=2)  # 1/8

        self.fpn3 = conv1x1(block_dims[2], block_dims[2])
        self.fpn2 = FPNBlock(block_dims[2], block_dims[1])
        self.fpn1 = FPNBlock(block_dims[1], block_dims[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x3_out = self.fpn3(x3)
        x2_out = self.fpn2(x3_out, x2)
        x1_out = self.fpn1(x2_out, x1)

        return [x3_out, x1_out]


class ResNetFPN_16_4(nn.Module):
    def __init__(self, initial_dim, block_dims):
        super().__init__()

        # downsample(1/2)
        self.enc0 = nn.Sequential(
            nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(initial_dim),
            nn.ReLU(inplace=True),
        )
        self.enc1 = EncoderBlock(initial_dim, block_dims[0], stride=1)  # 1/2
        self.enc2 = EncoderBlock(block_dims[0], block_dims[1], stride=2)  # 1/4
        self.enc3 = EncoderBlock(block_dims[1], block_dims[2], stride=2)  # 1/8
        self.enc4 = EncoderBlock(block_dims[2], block_dims[3], stride=2)  # 1/16

        self.fpn4 = conv1x1(block_dims[3], block_dims[3])
        self.fpn3 = FPNBlock(block_dims[3], block_dims[2])
        self.fpn2 = FPNBlock(block_dims[2], block_dims[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        x4_out = self.fpn4(x4)
        x3_out = self.fpn3(x4_out, x3)
        x2_out = self.fpn2(x3_out, x2)

        return [x4_out, x2_out]
