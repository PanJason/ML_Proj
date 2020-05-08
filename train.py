# Run train.py to train the model

import option
import numpy as np
import torch
import os
from torch.autograd import Variable

# 使用GPU?
gpu = [0] 
cuda_gpu = torch.cuda.is_available()

# para
epochs = 100
batch_size = 20
lr = 0.001
path = 'saved_model/net.pt'

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
    print('load model successfully\n')
except:
    print('cannot find model\n')


loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr)

for i in range(epochs):
    