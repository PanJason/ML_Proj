# Run train.py to train the model

import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models.resnet import *
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import ddpg
import option
import model
import data
import utility


def trainRibTracer(params):
    trainDataset = data.RibTraceDataset(
        params.data_dir, params.addi_path, params, True)
    valDataset = data.RibTraceDataset(
        params.val_data_set, params.val_addi_path, params)

    ribTracer = model.RibTracer(params)
    if params.continueModel != "None":
        d = torch.load(params.continueModel)
        ribTracer.load_state_dict(d)
    if params.useGPU:
        ribTracer = ribTracer.cuda()

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(
        ribTracer.parameters(), lr=params.learningRate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, verbose=True)

    def calcLoss(dataset):
        total = 0
        cnt = 0
        for batch, label in DataLoader(dataset,
                                       batch_size=params.batchSize,
                                       pin_memory=True,
                                       num_workers=2):
            if params.useGPU:
                batch = batch.cuda()
                label = label.cuda()
            with torch.no_grad():
                out = ribTracer(batch)
                loss = loss_fn(out, label)
            l = torch.sum(loss).cpu()
            total += l
            cnt += 1

        return float(total) / float(cnt)

    train_loss_rec = []
    val_loss_rec = []
    timer = utility.Timer()
    # torch.autograd.set_detect_anomaly(True)
    for epochID in range(1, params.numEpochs + 1):
        print(f"Start Epoch {epochID}")
        cnt = 0
        ribTracer.train()
        for batch, label in DataLoader(trainDataset,
                                       batch_size=params.batchSize,
                                       pin_memory=True,
                                       num_workers=2):
            optimizer.zero_grad()
            if params.useGPU:
                batch = batch.cuda()
                label = label.cuda()
            out = ribTracer(batch)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()

            if cnt % 100 == 0:
                print(f"Batch {cnt}: {loss} {timer()}")
            cnt += 1

        ribTracer.eval()
        trainloss = calcLoss(trainDataset)
        valloss = calcLoss(valDataset)
        train_loss_rec.append(trainloss)
        val_loss_rec.append(valloss)
        scheduler.step(trainloss)

        torch.save(ribTracer.state_dict(), os.path.join(
            params.model_path, "ribTracer.pt"))

        print(f"Epoch {epochID}: {timer()} Train: {trainloss} Val: {valloss}")


def trainVAE(params):
    trainDataset = data.VAEDataset(params.data_dir, params, True)
    valDataset = data.VAEDataset(params.val_data_set, params, True)

    VAEmodel = model.VAE(params)
    if params.useGPU:
        VAEmodel = VAEmodel.cuda()
    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        VAEmodel.parameters(), lr=params.learningRate)

    def calcLoss(dataset):
        total = 0
        cnt = 0
        for batch in DataLoader(dataset,
                                batch_size=params.batchSize,
                                pin_memory=True,
                                num_workers=4):
            if params.useGPU:
                batch = batch.cuda()
            with torch.no_grad():
                out = VAEmodel(batch)
                loss = loss_fn(out, batch)
            l = torch.sum(loss).cpu()
            total += l
            cnt += 1

        return float(total) / float(cnt)

    timer = utility.Timer()
    # torch.autograd.set_detect_anomaly(True)
    for epochID in range(1, params.numEpochs + 1):
        print(f"Start Epoch {epochID}")
        cnt = 0
        VAEmodel.train()
        trainloss = 0
        traintotal = 0
        for batch in DataLoader(trainDataset,
                                batch_size=params.batchSize,
                                pin_memory=True,
                                num_workers=4):
            optimizer.zero_grad()
            if params.useGPU:
                batch = batch.cuda()
            out = VAEmodel(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            trainloss += loss.detach().cpu().numpy()
            traintotal += 1
            optimizer.step()

            if cnt % 100 == 0:
                print(f"Batch {cnt}: {loss} {timer()}")
            cnt += 1

        VAEmodel.eval()
        trainloss /= traintotal
        # trainloss = calcLoss(trainDataset)
        valloss = calcLoss(valDataset)

        torch.save(VAEmodel.state_dict(), os.path.join(
            params.model_path, "VAE.pt"))
        torch.save(VAEmodel.conv.state_dict(), os.path.join(
            params.model_path, "ribTracerObserver.pt"))

        print(f"Epoch {epochID}: {timer()} Train: {trainloss} Val: {valloss}")


def trainRibTracerDDPG(params):
    trainDataset = data.RibTraceDDPGDataset(
        params.data_dir, params.addi_path, params, True)
    valDataset = data.RibTraceDDPGDataset(
        params.val_data_set, params.val_addi_path, params)

    ribTracer = ddpg.RibTracerDDPG(params)
    if params.continueModel != "None":
        ribTracer.loadWeights()
    if params.useGPU:
        ribTracer = ribTracer.cuda()

    timer = utility.Timer()
    for epochID in range(1, params.numEpochs+1):
        total_reward = 0
        total_len = 0
        cnt = 0
        ribTracer.train()
        for img, poly in trainDataset:
            reward, track = ribTracer.play(img, poly, poly[1] - poly[0], True)
            if cnt % 20 == 0:
                vloss = ribTracer.total_value_loss.cpu().numpy() / ribTracer.update_cnt
                ploss = ribTracer.total_policy_loss.cpu().numpy() / ribTracer.update_cnt
                print(
                    f"Batch {cnt}: Reward {reward} Len {len(track)} VLoss{vloss} PLoss{ploss} {timer()}")
            total_reward += reward
            total_len += len(track)
            cnt += 1
        ribTracer.eval()
        ribTracer.saveWeights()
        test_reward = 0
        test_len = 0
        test_cnt = 0
        for img, poly in valDataset:
            reward, track = ribTracer.play(img, poly, poly[1] - poly[0], False)
            test_reward += reward
            test_len += len(track)
            test_cnt += 1
        print(f"Epoch {epochID}: {timer()}")
        print(f"Train: {total_reward / cnt} {total_len/cnt}")
        print(f"Val: {test_reward / test_cnt} {test_len/test_cnt}")


def trainClassier(params):
    def validation(test_loader, model):
        correct = 0
        total_test = 0
        cnt = 0
        cross_entropy = 0
        model.eval()
        with torch.no_grad():
            for sample_batch in test_loader:
                images, labels = sample_batch
                if params.useGPU:
                    images, labels = Variable(
                        images.cuda()), Variable(labels.cuda())
                out = model.forward(images)
                loss = torch.nn.CrossEntropyLoss()(out, labels)

                _, pred = torch.max(out, 1)
                correct += (pred == labels).sum().item()
                cross_entropy += loss
                total_test += labels.size(0)
                cnt += 1

        return correct / total_test, cross_entropy / cnt

    train_data = ImageFolder(root=params.processed, transform=transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize(380),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(360),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))

    print(train_data.classes)
    train_loader = DataLoader(
        train_data, batch_size=params.batchSize, shuffle=True)
    testset = ImageFolder(root=params.val_data_set, transform=transforms.Compose([
        # transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))

    test_loader = DataLoader(testset, batch_size=4, shuffle=False)

    model = model.classifier(pre_train=True)

    if params.useGPU:
        print('gpu is available')
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    else:
        model = torch.nn.DataParallel(model)

    try:
        model.load_state_dict(torch.load(params.model_path))
        print('load model successfully')
    except:
        print('cannot find model')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), params.learningRate)

    for epoch in range(0, params.numEpochs):
        model.train()
        loss_sum = 0
        for batch_n, batch in enumerate(train_loader):
            start_time = time.time()
            inputs, labels = batch
            inputs, labels = Variable(inputs), Variable(labels)
            if params.useGPU:
                inputs, labels = Variable(
                    inputs.cuda()), Variable(labels.cuda())

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
                    epoch, params.numEpochs, batch_n, time.time() - start_time, loss_sum / 10, correct / labels.size(0)))
                loss_sum = 0

        acc, loss = validation(test_loader, model)
        print(
            'Epoch: [{}/{}], acc: {:.5f}, loss: {:.5f}'.format(epoch, params.numEpochs, acc, loss))
        if epoch % 5 == 4:
            torch.save(model.state_dict(), params.saved_path +
                       str(epoch)+'resnet34.pt')


def trainDetector(params):
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
            if mode == 0:  # 0 for train
                self.img_path = params.data_dir
                self.anno_path = params.addi_path
            elif mode == 1:  # 1 for validation
                self.img_path = params.val_data_set
                self.anno_path = params.addi_path
            with open(self.anno_path, 'r') as f:
                self.bbox = json.load(f)["bbox"]
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
                    if compute_iou(box1, box2) >= 0.5:
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
            if params.useGPU:
                inputs, labels, bboxes = \
                    Variable(inputs.cuda()), Variable(
                        labels.cuda()), Variable(bboxes.cuda())
            else:
                inputs, labels, bboxes = \
                    Variable(inputs), Variable(labels), Variable(bboxes)

            xProb, xAffine = model(inputs)
            a, b, c, d = torch.where(xProb == torch.max(xProb[:, 1, :, :]))
            print(xProb[a, b, c, d])
            affine = xAffine[0, :, c[0], d[0]]
            ymin = float(-0.5 * affine[0] - 0.5 *
                         affine[1] + affine[4] + c[0]) * 32
            xmin = float(-0.5 * affine[2] - 0.5 *
                         affine[3] + affine[5] + d[0]) * 32
            ymax = float(0.5 * affine[0] + 0.5 *
                         affine[1] + affine[4] + c[0]) * 32
            xmax = float(0.5 * affine[0] + 0.5 *
                         affine[3] + affine[5] + d[0]) * 32
            bbox_pred = torch.tensor([xmin, ymin, xmax, ymax])
            iou = compute_iou(bbox_pred, bboxes[0])
            iou_sum += iou
            cnt += 1

        return iou_sum / cnt

    train_data = PDdata()
    trian_loader = DataLoader(train_data, params.batchSize, shuffle=True)
    test_data = PDdata(mode=1, is_aug=False)
    test_loader = DataLoader(test_data, 1, shuffle=False)
    model = model.PlateDetector()

    if params.useGPU:
        print('gpu is available')
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    try:
        model.load_state_dict(torch.load(os.path.join(
            params.model_path, "99PD.pt")), strict=False)
        print('load pretrained model successfully')
    except:
        print('fail to load pretrained model')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learningRate, weight_decay= 0.0005)
    loss1 = nn.NLLLoss()
    loss2 = nn.L1Loss()
    for epoch in range(0, params.numEpochs):
        model.train()
        loss_sum = 0
        for batch_n, (inputs, labels, affines, _) in enumerate(trian_loader):
            start_time = time.time()
            if params.useGPU:
                inputs, labels, affines = \
                    Variable(inputs.cuda()), Variable(
                        labels.cuda()), Variable(affines.cuda())
            else:
                inputs, labels, affines = \
                    Variable(inputs), Variable(labels), Variable(affines)
            optimizer.zero_grad()
            xProb, xAffine = model(inputs)

            loc_loss = loss1(xProb, labels)
            mask = torch.unsqueeze(labels, 1)
            ymin = (-0.5 * xAffine[:, 0, :, :].unsqueeze(1) - 0.5*xAffine[:,
                                                                          1, :, :].unsqueeze(1)+xAffine[:, 4, :, :].unsqueeze(1))*mask
            xmin = (-0.5 * xAffine[:, 2, :, :].unsqueeze(1) - 0.5*xAffine[:,
                                                                          3, :, :].unsqueeze(1)+xAffine[:, 5, :, :].unsqueeze(1))*mask
            ymax = (0.5 * xAffine[:, 0, :, :].unsqueeze(1) + 0.5*xAffine[:,
                                                                         1, :, :].unsqueeze(1)+xAffine[:, 4, :, :].unsqueeze(1))*mask
            xmax = (0.5 * xAffine[:, 2, :, :].unsqueeze(1) + 0.5*xAffine[:,
                                                                         3, :, :].unsqueeze(1)+xAffine[:, 5, :, :].unsqueeze(1))*mask
            affine_box = torch.cat((xmin, ymin, xmax, ymax), dim=1)
            affine_loss = loss2(affine_box, affines)
            loss = loc_loss + affine_loss

            loss.backward()
            optimizer.step()
            loss_sum += loss
            if batch_n % 10 == 9:
                print('Epoch: [{}/{}], batch: {}, took: {:.3f}, loss: {:.5f}'.format(
                    epoch, params.numEpochs, batch_n, time.time() - start_time, loss_sum / 10))
                loss_sum = 0

        if epoch % 5 == 4:
            torch.save(model.state_dict(), params.saved_path+str(epoch)+'PD.pt')

        iou = validation(model, test_loader)
        print('Epoch: [{}/{}], aver_iou: {:.5}'.format(
            epoch, params.numEpochs, iou))


if __name__ == "__main__":
    params = option.read()

    train_functions = {
        "None": lambda x: None,
        "Classifier": trainClassier,
        "RibTracer": trainRibTracer,
        "RibTracerDDPG": trainRibTracerDDPG,
        "VAE": trainVAE
    }
    train_functions[params.trainModel](params)
