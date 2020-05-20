import torch
import torchvision
from torchvision import transforms
from model import classifier
from torch.autograd import Variable
from torch.utils.data import DataLoader


gpu = [0]
cuda_gpu = torch.cuda.is_available()

model_path = 'saved_model/classifier.pt'
test_path = 'data/val'

model = classifier(pre_train=False)

if cuda_gpu:
    print('gpu is available')
    model = torch.nn.DataParallel(model, device_ids=gpu).cuda()

try:
    model.load_state_dict(torch.load(model_path))
    print('load model successfully')
except:
    print('cannot find model')

model.eval()
testset = torchvision.datasets.ImageFolder(root=test_path, transform=transforms.Compose([
    # transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]))

test_loader = DataLoader(testset, batch_size=32, shuffle=False)

correct = 0
total_test = 0
cnt = 0
cross_entropy = 0

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


print('Acc: {:.3f}, Loss: {:.3f}'.format(correct / total_test, cross_entropy / cnt))