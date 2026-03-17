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


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, ch1x1: int, ch3x3_reduce: int, ch3x3: int,
                 ch5x5_reduce: int, ch5x5: int, pool_proj: int):
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3_reduce, kernel_size=1),
            BasicConv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5_reduce, kernel_size=1),
            BasicConv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        return self.fc2(x)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 1000, aux_logits: bool = True):
        super().__init__()
        self.aux_logits = aux_logits
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv3(self.conv2(x)))
        x = self.maxpool3(self.inception3b(self.inception3a(x)))
        x = self.inception4a(x)
        aux1: Optional[torch.Tensor] = self.aux1(x) if self.training and self.aux_logits else None
        x = self.inception4d(self.inception4c(self.inception4b(x)))
        aux2: Optional[torch.Tensor] = self.aux2(x) if self.training and self.aux_logits else None
        x = self.maxpool4(self.inception4e(x))
        x = self.inception5b(self.inception5a(x))
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(self.dropout(x))
        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x


def googlenet(num_classes: int = 1000) -> GoogLeNet:
    return GoogLeNet(num_classes=num_classes)
