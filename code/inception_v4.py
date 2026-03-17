import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
        )
        self.branch_pool = nn.MaxPool2d(3, stride=2)
        self.branch_conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, kernel_size=3),
        )
        self.branch3_pool = nn.MaxPool2d(3, stride=2)
        self.branch3_conv = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.cat([self.branch_pool(x), self.branch_conv(x)], 1)
        x = torch.cat([self.branch1(x), self.branch2(x)], 1)
        return torch.cat([self.branch3_pool(x), self.branch3_conv(x)], 1)


class InceptionA(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(384, 96, kernel_size=1),
        )
        self.branch2 = BasicConv2d(384, 96, kernel_size=1)
        self.branch3 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], 1)


class ReductionA(nn.Module):
    def __init__(self, in_channels: int, k: int, l: int, m: int, n: int):
        super().__init__()
        self.branch_pool = nn.MaxPool2d(3, stride=2)
        self.branch1 = BasicConv2d(in_channels, n, kernel_size=3, stride=2)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, k, kernel_size=1),
            BasicConv2d(k, l, kernel_size=3, padding=1),
            BasicConv2d(l, m, kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch_pool(x), self.branch1(x), self.branch2(x)], 1)


class InceptionB(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(1024, 128, kernel_size=1),
        )
        self.branch2 = BasicConv2d(1024, 384, kernel_size=1)
        self.branch3 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 224, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 7), padding=(0, 3)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], 1)


class ReductionB(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_pool = nn.MaxPool2d(3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch_pool(x), self.branch1(x), self.branch2(x)], 1)


class InceptionC(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(1536, 256, kernel_size=1),
        )
        self.branch2 = BasicConv2d(1536, 256, kernel_size=1)
        self.branch3_reduce = BasicConv2d(1536, 384, kernel_size=1)
        self.branch3a = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3b = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch4_reduce = BasicConv2d(1536, 384, kernel_size=1)
        self.branch4_conv = BasicConv2d(384, 448, kernel_size=(1, 3), padding=(0, 1))
        self.branch4_conv2 = BasicConv2d(448, 512, kernel_size=(3, 1), padding=(1, 0))
        self.branch4a = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch4b = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3_reduce(x)
        b3 = torch.cat([self.branch3a(b3), self.branch3b(b3)], 1)
        b4 = self.branch4_conv2(self.branch4_conv(self.branch4_reduce(x)))
        b4 = torch.cat([self.branch4a(b4), self.branch4b(b4)], 1)
        return torch.cat([self.branch1(x), self.branch2(x), b3, b4], 1)


class InceptionV4(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.stem = Stem()
        self.inception_a = nn.Sequential(*[InceptionA() for _ in range(4)])
        self.reduction_a = ReductionA(384, 192, 224, 256, 384)
        self.inception_b = nn.Sequential(*[InceptionB() for _ in range(7)])
        self.reduction_b = ReductionB()
        self.inception_c = nn.Sequential(*[InceptionC() for _ in range(3)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(self.dropout(x))


def inception_v4(num_classes: int = 1000) -> InceptionV4:
    return InceptionV4(num_classes=num_classes)
