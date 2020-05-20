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



class RibTracer(torch.nn.Module):
    def __init__(self, params):
        super(RibTracer, self).__init__()

        self.params = params

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv1", torch.nn.Conv2d(
            in_channels=1, out_channels=16, stride=2, kernel_size=5, padding=2))  # (16, 100)
        self.conv.add_module("relu1", torch.nn.ReLU())
        # self.conv.add_module("dropout2d1", torch.nn.Dropout2d(0.2))
        self.conv.add_module(
            "maxpool1", torch.nn.MaxPool2d(kernel_size=2))  # (16, 50)

        self.conv.add_module("conv2", torch.nn.Conv2d(
            in_channels=16, out_channels=32, stride=1, kernel_size=5))  # (32, 46)
        self.conv.add_module("relu2", torch.nn.ReLU())
        # self.conv.add_module("dropout2d2", torch.nn.Dropout2d(0.2))
        self.conv.add_module(
            "maxpool2", torch.nn.MaxPool2d(kernel_size=2))  # (32, 23)

        self.conv.add_module("conv3", torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4))  # (64, 20)
        self.conv.add_module("relu3", torch.nn.ReLU())
        # self.conv.add_module("dropout2d3", torch.nn.Dropout2d(0.2))
        self.conv.add_module(
            "maxpool3", torch.nn.MaxPool2d(kernel_size=2))  # (64, 10)

        self.conv.add_module("Flatten", torch.nn.Flatten())

        self.conv.add_module("fc1", torch.nn.Linear(
            64*int((self.params.regionSize / 16 - 5/2) ** 2), 50))  # (50)
        self.conv.add_module("relu4", torch.nn.ReLU())  # (50)
        self.conv.add_module("dropout1", torch.nn.Dropout(0.2))
        last_linear = torch.nn.Linear(50, 2)
        self.conv.add_module("fc2", last_linear)  # (2)
        self.conv.add_module("sigmoid", torch.nn.Sigmoid())

        self.apply(self.weights_init)
        torch.nn.init.xavier_uniform_(
            last_linear.weight, torch.nn.init.calculate_gain("sigmoid"))

    def forward(self, image):
        x = self.conv(image)
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
