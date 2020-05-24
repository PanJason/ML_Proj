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
from vertebraLandmark import spineFinder

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


def findSpine(params):
    network = spineFinder.Network(params)
    return network.test(params)


def findRibs(params):
    with open(os.path.join(params.median, "spine.json"), "r") as file:
        spines = json.load(file)
    ribTracer = model.RibTracer(params)
    d = torch.load(os.path.join(params.model_path, "ribTracer.pt"))
    ribTracer.load_state_dict(d)
    ribTracer.eval()

    files = os.listdir(params.data_set)
    result = {}
    for i, file in enumerate(files):
        file_id = file.split('.')[0]
        print(f"Tracing on image file ({i+1}/{len(files)}): {file}")
        img = Image.open(os.path.join(params.data_set, file))
        tracks = []
        for box in spines[file_id]:
            box = np.array(box)
            track = ribTrace(img, ribTracer, box[1], box[1] - box[2], params)
            track = [i.tolist() for i in track]
            tracks.append(track)
            track = ribTrace(img, ribTracer, box[2], box[2] - box[1], params)
            track = [i.tolist() for i in track]
            tracks.append(track)
        result[file_id] = tracks
    with open(os.path.join(params.median, "ribs.json"), "w") as file:
        json.dump(result, file, indent=4)
    return result


if __name__ == "__main__":
    params = option.read()
    if params.testTarget == "ribTracerDDPG":
        with open("additional_anno/additional_anno_val.json", "r") as file:
            anno = json.load(file)
        showRibTraceTrackDDPG("data/fracture/val_processed/101.png",
                              anno["poly"]["101"][1:], params)
    elif params.testTarget == "spine":
        view.showSpine("101")
    elif params.testTarget == "ribs":
        # findRibs(params)
        files = os.listdir(params.data_set)
        for f in files:
            view.showRibs(f.split('.')[0])
