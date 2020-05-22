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
