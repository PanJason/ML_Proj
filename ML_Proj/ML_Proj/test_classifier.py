# test classifier

import numpy as np
import torch
import torchvision
from torchvision import transforms
from model import Net
from torch.utils.data import DataLoader

model_path = 'saved_model/9net.pt'
test_path = 'data/val'

model = Net()

model.load_state_dict(torch.load(model_path))
print('load model successfully')
model.eval()

testset = torchvision.datasets.ImageFolder(root=test_path, transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
]))

test_loader = DataLoader(testset, batch_size=4, shuffle=False)

correct = 0
total_test = 0
cnt = 0
cross_entropy = 0

with torch.no_grad():
    for sample_batch in test_loader:
        images, labels = sample_batch

        out = model.forward(images)
        loss = torch.nn.CrossEntropyLoss()(out, labels)

        _, pred = torch.max(out, 1)
        correct += (pred == labels).sum().item()
        cross_entropy += loss
        total_test += labels.size(0)
        cnt += 1


print('Acc: {:.3f}, Loss: {:.3f}'.format(correct / total_test, cross_entropy / cnt))