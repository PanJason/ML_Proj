# Run inference.py to get result from one specified input

import math
import json
import os
import copy

import ddpg
import option
import data
import model
import view

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2 as cv


def ribTrace(img, ribTracer, start, direction, params):
    x = np.array(start)
    w, h = img.size
    img = torchvision.transforms.Grayscale(1)(img)
    img = torchvision.transforms.Pad(int(params.regionSize/2))(img)
    track = [copy.deepcopy(x)]
    for i in range(params.maxTrace):
        region = torchvision.transforms.functional.crop(
            img, int(x[1]), int(x[0]), params.regionSize, params.regionSize)
        region = torchvision.transforms.ToTensor()(region)
        region = region.view([1, 1, params.regionSize, params.regionSize])
        if params.useGPU:
            region = region.cuda()

        with torch.no_grad():
            delta = ribTracer(region).numpy()[0]
        delta[1] = delta[1] * 2.0 - 1.0
        delta = delta * params.regionSize * math.sqrt(2) / 2 * params.traceStep
        if np.dot(delta, direction) < 0:
            delta = -delta
        direction = delta
        x += delta
        x[0] = np.clip(x[0], 0, w)
        x[1] = np.clip(x[1], 0, h)
        # print(x, out[2])
        track.append(copy.deepcopy(x))
    return track


def ribTraceDDPG(img, ribTracer, start, direction, params):
    x = np.array(start)
    w, h = img.size
    img = torchvision.transforms.Pad(int(params.regionSize/2))(img)
    img = torchvision.transforms.ToTensor()(img)
    _, track = ribTracer.play(img, np.array([start]), direction, False)
    return track


def showRibTraceTrack(imgPath, polys, params):
    ribTracer = model.RibTracer(params)
    d = torch.load(os.path.join(params.model_path, "ribTracer.pt"))
    ribTracer.load_state_dict(d)
    ribTracer.eval()
    img = cv.imread(imgPath)
    img, wscale, hscale = view.scaling(img)
    origin_img = Image.open(imgPath)
    for i, poly in enumerate(polys):
        print(i+1)
        poly = np.array(poly)
        track = ribTrace(origin_img, ribTracer,
                         poly[0], poly[1]-poly[0], params)
        track = [(int(x[0] * wscale), int(x[1] * hscale)) for x in track]
        view.drawPoly(img, track, view.randomColor(), i+1)
    cv.imshow("Display", img)
    cv.waitKey(0)


def showRibTraceTrackDDPG(imgPath, polys, params):
    ribTracer = ddpg.RibTracerDDPG(params)
    ribTracer.loadWeights()
    ribTracer.eval()
    img = cv.imread(imgPath)
    img, wscale, hscale = view.scaling(img)
    origin_img = Image.open(imgPath)
    for i, poly in enumerate(polys):
        print(i+1)
        poly = np.array(poly)
        track = ribTraceDDPG(origin_img, ribTracer,
                             poly[0], poly[1]-poly[0], params)
        track = [(int(x[0] * wscale), int(x[1] * hscale)) for x in track]
        view.drawPoly(img, track, view.randomColor(), i+1)
    cv.imshow("Display", img)
    cv.waitKey(0)


if __name__ == "__main__":
    params = option.read()
    with open("additional_anno/additional_anno_val.json", "r") as file:
        anno = json.load(file)
    showRibTraceTrackDDPG("data/fracture/val_processed/101.png",
                          anno["poly"]["101"][1:], params)
