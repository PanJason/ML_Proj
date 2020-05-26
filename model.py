# Define model in this file

import numpy as np
import torch
import os
from torch.autograd import Variable
import option
from torchvision.models.resnet import *


sizen = 360


def classifier(pre_train):
    model = resnet34(pretrained=pre_train)
    model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    model.fc = torch.nn.Sequential(
        torch.nn.BatchNorm1d(512),
        torch.nn.Linear(512, 2)
    )
    return model


def makeRibTracerObserveNet(params):
    conv = torch.nn.Sequential()
    conv.add_module("conv1", torch.nn.Conv2d(
        in_channels=1, out_channels=16, stride=2, kernel_size=5, padding=2))  # (16, 100)
    conv.add_module("relu1", torch.nn.ReLU())
    # self.conv.add_module("dropout2d1", torch.nn.Dropout2d(0.2))
    conv.add_module(
        "maxpool1", torch.nn.MaxPool2d(kernel_size=2))  # (16, 50)

    conv.add_module("conv2", torch.nn.Conv2d(
        in_channels=16, out_channels=32, stride=1, kernel_size=5))  # (32, 46)
    conv.add_module("relu2", torch.nn.ReLU())
    # self.conv.add_module("dropout2d2", torch.nn.Dropout2d(0.2))
    conv.add_module(
        "maxpool2", torch.nn.MaxPool2d(kernel_size=2))  # (32, 23)

    conv.add_module("conv3", torch.nn.Conv2d(
        in_channels=32, out_channels=64, kernel_size=4))  # (64, 20)
    conv.add_module("relu3", torch.nn.ReLU())
    # self.conv.add_module("dropout2d3", torch.nn.Dropout2d(0.2))
    conv.add_module(
        "maxpool3", torch.nn.MaxPool2d(kernel_size=2))  # (64, 10)

    conv.add_module("Flatten", torch.nn.Flatten())

    conv.add_module("fc1", torch.nn.Linear(
        64*int((params.regionSize / 16 - 5/2) ** 2), 50))  # (50)
    conv.add_module("relu4", torch.nn.ReLU())  # (50)
    conv.add_module("dropout1", torch.nn.Dropout(0.2))

    return conv


class RibTracer(torch.nn.Module):
    def __init__(self, params):
        super(RibTracer, self).__init__()

        self.params = params

        self.conv = makeRibTracerObserveNet(params)

        self.predict = torch.nn.Sequential()
        last_linear = torch.nn.Linear(50, 2)
        self.predict.add_module("fc2", last_linear)  # (2)
        self.predict.add_module("sigmoid", torch.nn.Sigmoid())

        self.apply(self.weights_init)
        torch.nn.init.xavier_uniform_(
            last_linear.weight, torch.nn.init.calculate_gain("sigmoid"))

    def forward(self, image):
        x = self.conv(image)
        x = self.predict(x)
        return x

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.xavier_normal_(
                m.weight.data, torch.nn.init.calculate_gain("relu"))
        elif classname.find('BatchNorm') != -1 or classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(
                m.weight.data, torch.nn.init.calculate_gain("relu"))
            torch.nn.init.constant_(m.bias.data, 0)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding
    come from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution
    come from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    '''
    come from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    '''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PlateDetector(nn.Module):
    """
    This is the new plate bbox detection network based on YOLO.
    modified on resnet34
    Returns:
    * **xProb**: The probablity of the existence of the bounding box
    * **xAffine**: The Affine parameter of the base bounding box.
    """

    def __init__(self):
        super(PlateDetector, self).__init__()
        layers = [3, 4, 6, 3]
        self.norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1

        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.prob = nn.Conv2d(self.inplanes, 2, kernel_size=3, padding=1)
        self.affine = nn.Conv2d(self.inplanes, 6, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        xProb = self.prob(x)
        xProb = F.log_softmax(xProb, dim=1)

        xAffine = self.affine(x)
        return xProb, xAffine


class VAE(torch.nn.Module):
    def __init__(self, params):
        super(VAE, self).__init__()

        self.params = params

        self.conv = makeRibTracerObserveNet(params)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(50, 6400),
            torch.nn.ReLU()
        )
        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                64, 32, 5, 2, 2, dilation=2),  # 32, 24, 24
            # torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, 5, 2),  # 16, 49, 49
            # torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                16, 8, 4, 2),  # 8, 100, 100
            # torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                8, 1, 4, 2, 1),  # 1, 200, 200
            torch.nn.Sigmoid()
        )

    def forward(self, image):
        x = self.conv(image)
        x = self.linear(x)
        x = x.view((-1, 64, 10, 10))
        x = self.deconv(x)
        return x
