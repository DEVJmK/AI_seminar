import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels: int, pool_features: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5, padding=2),
        )
        self.branch3x3dbl = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_features, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1x1(x), self.branch5x5(x),
                          self.branch3x3dbl(x), self.branch_pool(x)], 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=2),
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch3x3(x), self.branch3x3dbl(x), self.branch_pool(x)], 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels: int, channels_7x7: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_channels, c7, kernel_size=1),
            BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.branch7x7dbl = nn.Sequential(
            BasicConv2d(in_channels, c7, kernel_size=1),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3)),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1x1(x), self.branch7x7(x),
                          self.branch7x7dbl(x), self.branch_pool(x)], 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 320, kernel_size=3, stride=2),
        )
        self.branch7x7x3 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(192, 192, kernel_size=3, stride=2),
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch3x3(x), self.branch7x7x3(x), self.branch_pool(x)], 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_reduce = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_reduce = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_conv = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1x1(x)
        b3 = self.branch3x3_reduce(x)
        b3 = torch.cat([self.branch3x3_a(b3), self.branch3x3_b(b3)], 1)
        b3dbl = self.branch3x3dbl_conv(self.branch3x3dbl_reduce(x))
        b3dbl = torch.cat([self.branch3x3dbl_a(b3dbl), self.branch3x3dbl_b(b3dbl)], 1)
        return torch.cat([b1, b3, b3dbl, self.branch_pool(x)], 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.conv0(self.avgpool(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


class InceptionV3(nn.Module):
    def __init__(self, num_classes: int = 1000, aux_logits: bool = True):
        super().__init__()
        self.aux_logits = aux_logits
        self.Conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.Conv2d_1a(x)
        x = self.Conv2d_2a(x)
        x = self.Conv2d_2b(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b(x)
        x = self.Conv2d_4a(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        aux: Optional[torch.Tensor] = self.AuxLogits(x) if self.training and self.aux_logits else None
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(self.dropout(x))
        if self.training and self.aux_logits:
            return x, aux
        return x


def inception_v2(num_classes: int = 1000) -> InceptionV3:
    return InceptionV3(num_classes=num_classes, aux_logits=False)

def inception_v3(num_classes: int = 1000) -> InceptionV3:
    return InceptionV3(num_classes=num_classes, aux_logits=True)
