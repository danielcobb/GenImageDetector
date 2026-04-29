import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from ml.classifiers.pytorch_base import PyTorchClassifier


CROP_SIZE = 224


def _tile_to_min_size(img: Image.Image, min_size: int = CROP_SIZE) -> Image.Image:
    """If either dimension is smaller than min_size, tile the image so it
    fits. Mirrors `translate_duplicate` from upstream NPR's GenImage recipe."""
    w, h = img.size
    if w >= min_size and h >= min_size:
        return img
    new_w = w * math.ceil(min_size / w)
    new_h = h * math.ceil(min_size / h)
    canvas = Image.new("RGB", (new_w, new_h))
    for x in range(0, new_w, w):
        for y in range(0, new_h, h):
            canvas.paste(img, (x, y))
    return canvas


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class _NPRResNet(nn.Module):
    """Truncated ResNet-50 (layer1+layer2 only) with NPR residual front-end.
    Vendored from chuangchuangtan/NPR-DeepfakeDetection."""

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=3)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * _Bottleneck.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * _Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * _Bottleneck.expansion),
            )
        layers = [_Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * _Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(_Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    @staticmethod
    def _interpolate_round_trip(img: torch.Tensor, factor: float) -> torch.Tensor:
        down = F.interpolate(
            img, scale_factor=factor, mode="nearest", recompute_scale_factor=True
        )
        return F.interpolate(
            down, scale_factor=1 / factor, mode="nearest", recompute_scale_factor=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Neighboring Pixel Relationship feature: x minus its 0.5x->2x reconstruction
        npr = x - self._interpolate_round_trip(x, 0.5)
        x = self.maxpool(self.relu(self.bn1(self.conv1(npr * 2.0 / 3.0))))
        x = self.layer2(self.layer1(x))
        x = self.avgpool(x).flatten(1)
        return self.fc1(x)


class NPRClassifier(PyTorchClassifier):
    """NPR detector — truncated ResNet50 trained on Neighboring Pixel
    Relationship residuals. Output is a single logit; sigmoid(logit) is
    the probability that the image is FAKE."""

    def __init__(self, model_path: str, device: str = None, quiet: bool = False):
        self.quiet = quiet
        super().__init__(model_path, device)

    def get_model_architecture(self) -> nn.Module:
        return _NPRResNet()

    def get_transforms(self):
        return transforms.Compose(
            [
                transforms.Lambda(_tile_to_min_size),
                transforms.CenterCrop(CROP_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def postprocess(self, output: torch.Tensor) -> float:
        # Single logit: sigmoid(logit) = P(fake). prob_real = sigmoid(-logit).
        logit = output.view(-1)[0].item()
        prob_real = torch.sigmoid(torch.tensor(-logit)).item()
        confidence = round(prob_real * 100, 1)
        if not self.quiet:
            print(f"[NPR Debug] Logit: {logit:.4f}, Prob(real): {prob_real:.4f}")
        return confidence
