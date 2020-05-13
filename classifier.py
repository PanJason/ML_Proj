import numpy as np
import torch
import os
from torch.autograd import Variable
sizen = 200


gpu = [0]
cuda_gpu = torch.cuda.is_available()
epochs = 100
batch_size = 20
lr = 0.001
path = 'saved_model/net.pt'

'''
def pre_train(region):
    newregion = np.zeros((sizen, sizen))
    for i in range(sizen):
        for j in range(sizen):
            if i > 0:
                newregion[i][j] = max(newregion[i][j], abs(region[i-1][j]- region[i][j]))
            if j > 0:
                newregion[i][j] = max(newregion[i][j], abs(region[i][j-1]- region[i][j]))
            if i < sizen - 1:
                newregion[i][j] = max(newregion[i][j], abs(region[i+1][j]- region[i][j]))
            if j < sizen - 1:
                newregion[i][j] = max(newregion[i][j], abs(region[i][j+1]- region[i][j]))
    return newregion
'''

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv1", torch.nn.Conv2d(in_channels = 1, out_channels=32, kernel_size=5)) # (-1, 32, sizen-4, sizen-4)
        self.conv.add_module("relu1", torch.nn.ReLU())
        self.conv.add_module("maxpool1", torch.nn.MaxPool2d(kernel_size=2)) # (-1, 32, sizen/2 - 2, sizen/2 - 2)

        self.conv.add_module("conv2", torch.nn.Conv2d(in_channels= 32, out_channels=64, kernel_size=5)) # (-1, 64, sizen/2-6, sizen/2-6)
        self.conv.add_module("relu2", torch.nn.ReLU())
        self.conv.add_module("maxpool2", torch.nn.MaxPool2d(kernel_size=2)) # (-1, 64, sizen/4-3, sizen/4-3)

        self.conv.add_module("conv3", torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)) # (-1, 64, sizen/4 - 5, sizen/4-5)
        self.conv.add_module("relu3", torch.nn.ReLU())
        self.conv.add_module("maxpool3", torch.nn.MaxPool2d(kernel_size=2)) # (-1, 64, sizen/8 - 3, sizen/8-3)

        self.fc = torch.nn.Sequential()
        self.fc.add_module("dropout1", torch.nn.Dropout(0.5))
        self.fc.add_module("fc1", torch.nn.Linear(64*(sizen/8-3)*(sizen/8-3), 50)) # (-1, 50)
        self.fc.add_module("relu3", torch.nn.ReLU())
        self.fc.add_module("dropout2", torch.nn.Dropout(0.5))
        self.fc.add_module("fc2", torch.nn.Linear(50, 1))
        self.fc.add_module("sigmoid", torch.nn.Sigmoid())

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 64*(sizen/8-3)*(sizen/8-3))
        return self.fc.forward(x)


def train(model, loss, optimizer, x, y):
    optimizer.zero_grad()
    fx = model.forward(x)
    output = loss.forward(fx, y)

    output.backward()
    optimizer.step()

    return output

def predict(model, x):
    return model.forward(x)


'''
data prepare...
'''

net = Net()
try:    
    net.load_state_dict(torch.load(path))
    print('load model successfully')
except:
    print('cannot find model')


loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr)

for i in range(epochs):
    


