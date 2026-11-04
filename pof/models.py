"""Models: CIFAR-style ResNet-20 (He et al. 2016, sec. 4.2), a slim VGG-11, and a tiny CNN for smoke runs.

ResNet-20 uses GroupNorm instead of BatchNorm. BatchNorm running statistics are
shared state that FedAvg averages awkwardly under non-IID clients and that leak
membership through a channel unrelated to the weights; GroupNorm keeps the model
a pure function of its parameters, which is what every verifier here assumes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = _norm(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = _norm(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), _norm(cout))

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        return F.relu(self.n2(self.conv2(out)) + self.short(x))


class ResNet20(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 3, width: int = 16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, width, 3, 1, 1, bias=False),
                                  _norm(width), nn.ReLU())
        layers, cin = [], width
        for stage, cout in enumerate((width, 2 * width, 4 * width)):
            for b in range(3):
                layers.append(BasicBlock(cin, cout, 2 if (b == 0 and stage > 0) else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class VGG11(nn.Module):
    """VGG-11 (Simonyan and Zisserman 2015, configuration A) for small images: a plain
    stack of 3x3 convolutions and max-pools with no skip connections, so a second
    architecture family next to ResNet-20. GroupNorm for the same reason as ResNet-20.

    Channels are VGG's divided by 64/width (width 16 -> 16..128 instead of 64..512),
    which keeps the compute close to ResNet-20's so a run fits a Kaggle session. Four
    max-pools instead of five, then global average pooling, so 28x28 inputs also work.
    """
    def __init__(self, num_classes: int, in_ch: int = 3, width: int = 16):
        super().__init__()
        w = width
        plan = [w, "M", 2 * w, "M", 4 * w, 4 * w, "M", 8 * w, 8 * w, "M", 8 * w, 8 * w]
        layers, cin = [], in_ch
        for v in plan:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(cin, v, 3, padding=1, bias=False), _norm(v), nn.ReLU()]
                cin = v
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 3, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * width, num_classes))

    def forward(self, x):
        return self.net(x)


def build_model(cfg: dict, num_classes: int, in_ch: int = 3) -> nn.Module:
    m = cfg["model"]
    if m["name"] == "resnet20":
        return ResNet20(num_classes, in_ch, m.get("width", 16))
    if m["name"] == "vgg11":
        return VGG11(num_classes, in_ch, m.get("width", 16))
    if m["name"] == "smallcnn":
        return SmallCNN(num_classes, in_ch, m.get("width", 16))
    raise ValueError(f"unknown model '{m['name']}'")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
