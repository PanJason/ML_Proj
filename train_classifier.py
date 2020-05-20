# Run train.py to train the model

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import classifier
import time

# use GPU?
gpu = [0] 
cuda_gpu = torch.cuda.is_available()

# parameter
epochs = 300
pre_epoch = 0
batch_size = 32
lr = 0.01
model_path = 'saved_model/classifier.pt'
save_path = 'saved_model/'
data_path = 'data/train'


train_data = ImageFolder(root=data_path, transform=transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize(380),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(360),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]))

print(train_data.classes)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

model = classifier(pre_train=True)

if cuda_gpu:
    print('gpu is available')
    model = torch.nn.DataParallel(model, device_ids=gpu).cuda()

try:
    model.load_state_dict(torch.load(model_path))
    print('load model successfully')
except:
    print('cannot find model')


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

model.train()

for epoch in range(pre_epoch, epochs):
    loss_sum = 0
    for batch_n, batch in enumerate(train_loader):
        start_time = time.time()
        inputs, labels = batch
        if cuda_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        if batch_n % 10 == 9:
            _, pred = torch.max(outputs, 1)
            correct = (pred == labels).sum().item()
            print('Epoch: [{}/{}], batch: {}, took: {:.3f}, loss: {:.5f}, Acc: {:.5f}'.format(
                epoch, epochs, batch_n, time.time() - start_time, loss_sum / 10, correct / labels.size(0)))
            loss_sum = 0

    if epoch % 5 == 4:
        torch.save(model.state_dict(), save_path+str(epoch)+'classifier.pt')

