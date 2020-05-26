# Run train.py to train the model

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import time
from torchvision.models.resnet import *
from model import classifier

# use GPU?
gpu = [0] 
cuda_gpu = torch.cuda.is_available()

# parameter
epochs = 100
pre_epoch = 0
batch_size = 32
lr = 0.01
model_path = 'saved_model/classifier.pt'
save_path = 'saved_model/'
data_path = 'data/train'
test_path = 'data/val'


def validation(test_loader, model):
    correct = 0
    total_test = 0
    cnt = 0
    cross_entropy = 0
    model.eval()
    with torch.no_grad():
        for sample_batch in test_loader:
            images, labels = sample_batch
            if cuda_gpu:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            out = model.forward(images)
            loss = torch.nn.CrossEntropyLoss()(out, labels)

            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item()
            cross_entropy += loss
            total_test += labels.size(0)
            cnt += 1

    return correct / total_test, cross_entropy / cnt


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
testset = ImageFolder(root=test_path, transform=transforms.Compose([
    # transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]))

test_loader = DataLoader(testset, batch_size=4, shuffle=False)


model = classifier(pre_train=True)

if cuda_gpu:
    print('gpu is available')
    model = torch.nn.DataParallel(model, device_ids=gpu).cuda()
else:
    model = torch.nn.DataParallel(model)

try:
    model.load_state_dict(torch.load(model_path))
    print('load model successfully')
except:
    print('cannot find model')


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

for epoch in range(pre_epoch, epochs):
    model.train()
    loss_sum = 0
    for batch_n, batch in enumerate(train_loader):
        start_time = time.time()
        inputs, labels = batch
        inputs, labels = Variable(inputs), Variable(labels)
        if cuda_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

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

    acc, loss = validation(test_loader, model)
    print('Epoch: [{}/{}], acc: {:.5f}, loss: {:.5f}'.format(epoch, epochs, acc, loss))
    if epoch % 5 == 4:
        torch.save(model.state_dict(), save_path+str(epoch)+'resnet34.pt')

