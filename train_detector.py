from model import PlateDetector
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import numpy as np
import cv2
from PIL import Image
import time
from torch.autograd import Variable
from torch import nn
from torchvision import transforms

gpu = [0]
cuda_gpu = torch.cuda.is_available()
sizen = 360
threshold = 0.3
pre_epoch = 100
epochs = 200
batch_size = 16
lr = 0.001
weight_decay = 0.0005
save_path = 'saved_model/'
preModel_path = 'saved_model/99PD.pt'


def compute_iou(box1, box2):
    '''Compute the intersection over union of two boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [4].
      box2: (tensor) bounding boxes, sized [4].
    Return:
      (tensor) iou.
    '''
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    area1 = ((xmax1 - xmin1) * (ymax1 - ymin1))
    area2 = ((xmax2 - xmin2) * (ymax2 - ymin2))
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou


class PDdata(torch.utils.data.Dataset):
    def __init__(self, mode=0, is_aug=True):
        self.mode = mode
        self.is_aug = is_aug
        if mode == 0: # 0 for train
            self.img_path = 'detectorData/train/img'
            self.anno_path = 'detectorData/train/bbox.json'
        elif mode == 1: # 1 for validation
            self.img_path = 'detectorData/val/img'
            self.anno_path = 'detectorData/val/bbox.json'
        with open(self.anno_path, 'r') as f:
            self.bbox = json.load(f)
        self.map = [i for i in self.bbox.keys()]

    def __len__(self):
        return len(self.bbox)

    def __getitem__(self, item):
        '''
        return:
            img: (C, H, W)
            labels (H, W)
            affine (4, H, W) --- dim0: [xmin, ymin, xmax, ymax]
            bbax [xmin, ymin, xmax, ymax]
        '''
        img = cv2.imread(self.img_path+'/'+self.map[item]+'.png')
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.is_aug:
            img = transforms.ColorJitter(contrast=1)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
        # img = torch.unsqueeze(img, 0)

        # compute labels and Affine for img
        x, y, width, height = self.bbox[self.map[item]]
        bbox = torch.tensor([x, y, x + width, y + height])
        labels = torch.LongTensor(12, 12).zero_()
        affine = torch.zeros(4, 12, 12)
        x = x / 32
        y = y / 32
        width = width / 32
        height = height / 32
        box2 = torch.tensor([x, y, x+width, y+height])
        for i in range(12):
            for j in range(12):
                box1 = torch.zeros(4)
                box1[0] = j - width/2
                box1[1] = i - height/2
                box1[2] = j + width/2
                box1[3] = i + height/2
                if compute_iou(box1, box2) >= threshold:
                    labels[i][j] = 1.0
                    affine[0][i][j] = x - j
                    affine[1][i][j] = y - i
                    affine[2][i][j] = x + width - j
                    affine[3][i][j] = y + height - i

        return img, labels, affine, bbox


def validation(model, test_loader):
    model.eval()

    iou_sum = 0
    cnt = 0
    for batch_n, (inputs, labels, _, bboxes) in enumerate(test_loader):
        if cuda_gpu:
            inputs, labels, bboxes = \
                Variable(inputs.cuda()), Variable(labels.cuda()), Variable(bboxes.cuda())
        else:
            inputs, labels, bboxes = \
                Variable(inputs), Variable(labels), Variable(bboxes)

        xProb, xAffine = model(inputs)
        a, b, c, d = torch.where(xProb == torch.max(xProb[:, 1, :, :]))
        print(xProb[a, b, c, d])
        affine = xAffine[0, :, c[0], d[0]]
        ymin = float(-0.5 * affine[0] - 0.5 * affine[1] + affine[4] + c[0]) * 32
        xmin = float(-0.5 * affine[2] - 0.5 * affine[3] + affine[5] + d[0]) * 32
        ymax = float(0.5 * affine[0] + 0.5 * affine[1] + affine[4] + c[0]) * 32
        xmax = float(0.5 * affine[0] + 0.5 * affine[3] + affine[5] + d[0]) * 32
        bbox_pred = torch.tensor([xmin, ymin, xmax, ymax])
        iou = compute_iou(bbox_pred, bboxes[0])
        iou_sum += iou
        cnt += 1

    return iou_sum / cnt


train_data = PDdata()
trian_loader = DataLoader(train_data, batch_size, shuffle=True)
test_data = PDdata(mode=1, is_aug=False)
test_loader = DataLoader(test_data, 1, shuffle=False)
model = PlateDetector()

if cuda_gpu:
    print('gpu is available')
    model = torch.nn.DataParallel(model, device_ids=gpu).cuda()

try:
    model.load_state_dict(torch.load(preModel_path), strict=False)
    print('load pretrained model successfully')
except:
    print('fail to load pretrained model')


optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss1 = nn.NLLLoss()
loss2 = nn.L1Loss()
for epoch in range(pre_epoch, epochs):
    model.train()
    loss_sum = 0
    for batch_n, (inputs, labels, affines, _) in enumerate(trian_loader):
        start_time = time.time()
        if cuda_gpu:
            inputs, labels, affines = \
                Variable(inputs.cuda()), Variable(labels.cuda()), Variable(affines.cuda())
        else:
            inputs, labels, affines = \
                Variable(inputs), Variable(labels), Variable(affines)
        optimizer.zero_grad()
        xProb, xAffine = model(inputs)

        loc_loss = loss1(xProb, labels)
        mask = torch.unsqueeze(labels, 1)
        ymin = (-0.5 * xAffine[:,0,:,:].unsqueeze(1) -0.5*xAffine[:,1,:,:].unsqueeze(1)+xAffine[:,4,:,:].unsqueeze(1))*mask
        xmin = (-0.5 * xAffine[:,2,:,:].unsqueeze(1) -0.5*xAffine[:,3,:,:].unsqueeze(1)+xAffine[:,5,:,:].unsqueeze(1))*mask
        ymax = (0.5 * xAffine[:,0,:,:].unsqueeze(1) + 0.5*xAffine[:,1,:,:].unsqueeze(1)+xAffine[:,4,:,:].unsqueeze(1))*mask
        xmax = (0.5 * xAffine[:,2,:,:].unsqueeze(1) +0.5*xAffine[:,3,:,:].unsqueeze(1)+xAffine[:,5,:,:].unsqueeze(1))*mask
        affine_box = torch.cat((xmin, ymin, xmax, ymax), dim=1)
        affine_loss = loss2(affine_box, affines)
        loss = loc_loss + affine_loss

        loss.backward()
        optimizer.step()
        loss_sum += loss
        if batch_n % 10 == 9:
            print('Epoch: [{}/{}], batch: {}, took: {:.3f}, loss: {:.5f}'.format(
                epoch, epochs, batch_n, time.time() - start_time, loss_sum / 10))
            loss_sum = 0

    if epoch % 5 == 4:
        torch.save(model.state_dict(), save_path+str(epoch)+'PD.pt')

    iou = validation(model, test_loader)
    print('Epoch: [{}/{}], aver_iou: {:.5}'.format(
                epoch, epochs, iou))