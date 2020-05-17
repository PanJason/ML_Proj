# Run train.py to train the model

import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import option
import model
import data
import utility


def trainClassier(params):
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

    net = model.Net()
    try:
        net.load_state_dict(torch.load(path))
        print('load model successfully\n')
    except:
        print('cannot find model\n')

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)

    for i in range(epochs):
        pass


def trainRibTracer(params):
    trainDataset = data.RibTraceDataset(
        params.data_set, params.addi_path, params, True)
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


if __name__ == "__main__":
    params = option.read()

    train_functions = {
        "None": lambda x: None,
        "Classifier": trainClassier,
        "RibTracer": trainRibTracer
    }
    train_functions[params.trainModel](params)
