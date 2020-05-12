# Define model in this file

import numpy as np
import torch
import os
from torch.autograd import Variable
import option

sizen = 200


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv1", torch.nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5))  # (-1, 32, sizen-4, sizen-4)
        self.conv.add_module("relu1", torch.nn.ReLU())
        self.conv.add_module("maxpool1", torch.nn.MaxPool2d(
            kernel_size=2))  # (-1, 32, sizen/2 - 2, sizen/2 - 2)

        self.conv.add_module("conv2", torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5))  # (-1, 64, sizen/2-6, sizen/2-6)
        self.conv.add_module("relu2", torch.nn.ReLU())
        self.conv.add_module("maxpool2", torch.nn.MaxPool2d(
            kernel_size=2))  # (-1, 64, sizen/4-3, sizen/4-3)

        self.conv.add_module("conv3", torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3))  # (-1, 64, sizen/4 - 5, sizen/4-5)
        self.conv.add_module("relu3", torch.nn.ReLU())
        self.conv.add_module("maxpool3", torch.nn.MaxPool2d(
            kernel_size=2))  # (-1, 64, sizen/8 - 3, sizen/8-3)

        self.fc = torch.nn.Sequential()
        self.fc.add_module("dropout1", torch.nn.Dropout(0.5))
        self.fc.add_module("fc1", torch.nn.Linear(
            64*(sizen/8-3)*(sizen/8-3), 50))  # (-1, 50)
        self.fc.add_module("relu3", torch.nn.ReLU())
        self.fc.add_module("dropout2", torch.nn.Dropout(0.5))
        self.fc.add_module("fc2", torch.nn.Linear(50, 1))
        self.fc.add_module("sigmoid", torch.nn.Sigmoid())

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 64*(sizen/8-3)*(sizen/8-3))
        return self.fc.forward(x)


class RibTracer(torch.nn.Module):
    def __init__(self, params):
        super(RibTracer, self).__init__()

        self.params = params

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv1", torch.nn.Conv2d(
            in_channels=1, out_channels=32, stride=2, kernel_size=5, padding=2))  # (32, 100)
        self.conv.add_module("batchNorm1", torch.nn.BatchNorm2d(32))
        self.conv.add_module(
            "maxpool1", torch.nn.MaxPool2d(kernel_size=2))  # (32, 50)

        self.conv.add_module("conv2", torch.nn.Conv2d(
            in_channels=32, out_channels=64, stride=1, kernel_size=5))  # (64, 46)
        self.conv.add_module("batchNorm2", torch.nn.BatchNorm2d(64))
        self.conv.add_module("maxpool2", torch.nn.MaxPool2d(
            kernel_size=2))  # (64, 23)

        self.conv.add_module("conv3", torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=4))  # (64, 20)
        self.conv.add_module("relu3", torch.nn.ReLU())
        self.conv.add_module("maxpool3", torch.nn.MaxPool2d(
            kernel_size=2))  # (64, 10)

        self.conv.add_module("Flatten", torch.nn.Flatten())

        self.conv.add_module("dropout1", torch.nn.Dropout(0.2))
        self.conv.add_module("fc1", torch.nn.Linear(
            64*int((self.params.regionSize / 16 - 5/2) ** 2), 50))  # (50)
        self.conv.add_module("relu4", torch.nn.ReLU())  # (50)
        self.conv.add_module("fc2", torch.nn.Linear(50, 3))  # (2)
        self.conv.add_module("tanh", torch.nn.Tanh())

        self.add_module("Conv", self.conv)

    def forward(self, image):
        x = self.conv.forward(image)
        return x
