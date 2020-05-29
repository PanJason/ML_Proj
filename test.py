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
import detector
import IOU
from utility import Timer
from vertebraLandmark import spineFinder
import yolo_for_chest

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2 as cv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def ribTrace(img, ribTracer, start, direction, params):
    x = np.array(start)
    w, h = img.size
    img = torchvision.transforms.Grayscale(1)(img)
    img = torchvision.transforms.Pad(int(params.regionSize / 2))(img)
    track = [copy.deepcopy(x)]
    for i in range(params.maxTrace):
        region = torchvision.transforms.functional.crop(
            img, int(x[1]), int(x[0]), params.regionSize, params.regionSize)
        region = torchvision.transforms.ToTensor()(region)
        region = region.view([1, 1, params.regionSize, params.regionSize])
        # if params.useGPU:
        # region = region.cuda()

        with torch.no_grad():
            delta = ribTracer(region)
        delta = delta.cpu().numpy()[0]
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
    img = torchvision.transforms.Pad(int(params.regionSize / 2))(img)
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
        print(i + 1)
        poly = np.array(poly)
        track = ribTrace(origin_img, ribTracer,
                         poly[0], poly[1] - poly[0], params)
        track = [(int(x[0] * wscale), int(x[1] * hscale)) for x in track]
        view.drawPoly(img, track, view.randomColor(), i + 1)
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
        print(i + 1)
        poly = np.array(poly)
        track = ribTraceDDPG(origin_img, ribTracer,
                             poly[0], poly[1] - poly[0], params)
        track = [(int(x[0] * wscale), int(x[1] * hscale)) for x in track]
        view.drawPoly(img, track, view.randomColor(), i + 1)
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
    # if params.useGPU:
    # ribTracer = ribTracer.cuda()

    files = os.listdir(params.data_set)
    result = {}
    for i, file in enumerate(files):
        file_id = file.split('.')[0]
        print(f"Tracing on image file ({i + 1}/{len(files)}): {file}")
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


def findFractureClassifier(params):
    with open(os.path.join(params.median, "ribs.json"), "r") as file:
        ribs = json.load(file)
    dataset = data.ClassifierTestDataset(params.data_set, ribs, params)

    # classifier = model.classifier(False, os.path.join(
    # params.model_path, "resnet34-333f7ec4.pth"))
    classifier = model.classifier(False)
    classifier = torch.nn.DataParallel(classifier, device_ids=[0])
    if params.useGPU:
        classifier = classifier.cuda()
    d = torch.load(os.path.join(params.model_path, "classifier.pt"))
    # d1 = torch.load()
    # d.update(d1)
    classifier.load_state_dict(d)

    classifier.eval()
    detected = []
    cnt = 0
    timer = Timer()
    for batch, centers, imgIDs in torch.utils.data.DataLoader(dataset,
                                                              batch_size=params.batchSize,
                                                              pin_memory=True,
                                                              num_workers=0):
        cnt += 1
        if params.useGPU:
            batch = batch.cuda()
        with torch.no_grad():
            output = classifier(batch)
            output = torch.nn.functional.softmax(output, 1)
        output = output.cpu().numpy()
        centers = centers.numpy()
        imgIDs = imgIDs.numpy()
        for i in range(output.shape[0]):
            out = output[i]
            if out[1] > params.detectThreshold:
                detected.append((centers[i], imgIDs[i], out[1]))
        if cnt % 100 == 0:
            print(f"Batch {cnt} {timer()}")

    with open(params.anno_path, "r") as file:
        anno = json.load(file)
    anno["annotations"] = []
    regionSize = params.detectRegionSize
    for i, d in enumerate(detected):
        center, imgID, score = d
        anno["annotations"].append(
            {
                "bbox": [
                    float(center[0]) - regionSize / 2,
                    float(center[1]) - regionSize / 2,
                    regionSize,
                    regionSize
                ],
                "id": i,
                "image_id": int(imgID),
                "score": float(score)
            }
        )
    with open(os.path.join(params.median, "detection.json"), "w") as file:
        json.dump(anno, file, indent=4)


def findFractureYolo(params):
    with open(os.path.join(params.median, "ribs.json"), "r") as file:
        ribs = json.load(file)
    dataset = data.ClassifierTestDataset(params.data_set, ribs, params)
    fd = detector.fracture_detector()
    if params.useGPU:
        fd.model = fd.model.cuda()
    detected = []
    cnt = 0
    timer = Timer()
    for batch, centers, imgIDs in torch.utils.data.DataLoader(dataset,
                                                              batch_size=params.batchSize,
                                                              pin_memory=True,
                                                              num_workers=0):
        cnt += 1
        input = [batch[i].numpy().transpose(1, 2, 0)
                 for i in range(batch.shape[0])]
        output = fd.detectFracture(img=input)
        imgIDs = imgIDs.numpy()
        centers = centers.numpy()
        for i in range(batch.shape[0]):
            if i in output and output[i]['score'] > params.detectThreshold:
                detected.append((centers[i], imgIDs[i], output[i]))
        if cnt % 100 == 0:
            print(f"Batch {cnt} {timer()}")

    with open(params.anno_path, "r") as file:
        anno = json.load(file)
    anno["annotations"] = []
    regionSize = params.detectRegionSize
    for i, d in enumerate(detected):
        center, imgID, output = d
        anno["annotations"].append(
            {
                "bbox": [
                    str(float(center[0]) - regionSize /
                        2 + output['bbox'][0] * regionSize),
                    str(float(center[1]) - regionSize /
                        2 + output['bbox'][1] * regionSize),
                    str(output['bbox'][2] * regionSize),
                    str(output['bbox'][3] * regionSize)
                ],
                "id": i,
                "image_id": int(imgID),
                "score": str(output['score']),
            }
        )
    with open(os.path.join(params.median, "detection.json"), "w") as file:
        json.dump(anno, file, indent=4)


def findFractureChestDivide(params):
    with open(os.path.join(params.median, "ribs.json"), "r") as file:
        ribs = json.load(file)
    dataset = data.ChestDivideDataset(
        params.data_set, os.path.join(params.median, "chest.json"), params)
    fd = detector.fracture_detector()
    if params.useGPU:
        fd.model = fd.model.cuda()
    detected = []
    cnt = 0
    timer = Timer()
    for batch, centers, imgIDs in torch.utils.data.DataLoader(dataset,
                                                              batch_size=params.batchSize,
                                                              pin_memory=True,
                                                              num_workers=0):
        cnt += 1
        input = [batch[i].numpy().transpose(1, 2, 0)
                 for i in range(batch.shape[0])]
        output = fd.detectFracture(
            img=input, conf_thres=params.conf_thresh, iou_thres=params.seg_thresh)
        imgIDs = imgIDs.numpy()
        centers = centers.numpy()
        for i in range(batch.shape[0]):
            if i in output and output[i]['score'] > params.detectThreshold:
                detected.append((centers[i], imgIDs[i], output[i]))
        if cnt % 100 == 0:
            print(f"Batch {cnt} {timer()}")

    with open(params.anno_path, "r") as file:
        anno = json.load(file)
    anno["annotations"] = []
    regionSize = params.detectRegionSize
    for i, d in enumerate(detected):
        center, imgID, output = d
        anno["annotations"].append(
            {
                "bbox": [
                    str(float(center[0]) - regionSize /
                        2 + output['bbox'][0] * regionSize),
                    str(float(center[1]) - regionSize /
                        2 + output['bbox'][1] * regionSize),
                    str(output['bbox'][2] * regionSize),
                    str(output['bbox'][3] * regionSize)
                ],
                "id": i,
                "image_id": int(imgID),
                "score": str(output['score']),
            }
        )
    with open(os.path.join(params.median, "detection.json"), "w") as file:
        json.dump(anno, file, indent=4)


def findChest(params):
    yolo_for_chest.test_chest(params)


def bboxIntersect(A, B):
    ax1 = A[0]
    ay1 = A[1]
    ax2 = A[0] + A[2]
    ay2 = A[1] + A[3]
    bx1 = B[0]
    by1 = B[1]
    bx2 = B[0] + B[2]
    by2 = B[1] + B[3]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return [x1, y1, x2 - x1, y2 - y1]


def bboxUnion(A, B):
    ax1 = A[0]
    ay1 = A[1]
    ax2 = A[0] + A[2]
    ay2 = A[1] + A[3]
    bx1 = B[0]
    by1 = B[1]
    bx2 = B[0] + B[2]
    by2 = B[1] + B[3]
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    x2 = max(ax2, bx2)
    y2 = max(ay2, by2)
    return [x1, y1, x2 - x1, y2 - y1]


def postProcess(params, merge=False):
    with open(os.path.join(params.median, "detection.json"), "r") as file:
        anno = json.load(file)
    intersect = None
    lastID = -1
    sum_score = 0.0
    cnt = 0

    # imgBoxes = {}
    # for box in anno["annotations"]:
    #     bbox = box["bbox"]
    #     box["bbox"] = list(map(float, bbox))
    #     imgID = box["image_id"]
    #     box["score"] = float(box["score"])
    #     if imgID not in imgBoxes:
    #         imgBoxes[imgID] = [box]
    #     else:
    #         imgBoxes[imgID].append(box)

    # if merge:
    #     result = []
    #     for _, imgBox in imgBoxes.items():
    #         boxes = {}
    #         for x, box in enumerate(imgBox):
    #             bbox = box["bbox"]
    #             imgID = box["image_id"]
    #             score = float(box["score"])

    #             res = bbox
    #             remove_ids = []
    #             for i, b in boxes.items():
    #                 intersect = bboxIntersect(res, b[0])
    #                 if intersect[2] > 0 and intersect[3] > 0:
    #                     res = intersect
    #                     score = min(b[2], score)
    #                     remove_ids.append(i)
    #             for i in remove_ids:
    #                 del boxes[i]
    #             boxes[x] = (res, imgID, score)
    #         for i, b in boxes.items():
    #             result.append(b)
    # else:
    #     result = [(imgBox["bbox"], imgBox["image_id"], imgBox["score"])
    #               for imgBox in anno["annotations"]]

    result = []
    for box in anno["annotations"]:
        bbox = box["bbox"]
        bbox = list(map(float, bbox))
        imgID = box["image_id"]
        score = float(box["score"])
        if lastID != imgID:
            if intersect is not None:
                result.append((intersect, lastID, sum_score / cnt))
            lastID = imgID
            intersect = bbox
            sum_score = score
            cnt = 1
        else:
            I = bboxIntersect(bbox, intersect)
            if I[2] > 0 and I[3] > 0:
                intersect = I
                cnt += 1
                sum_score += score
            else:
                result.append((intersect, lastID, sum_score / cnt))
                intersect = bbox
                sum_score = score
                cnt = 1
    if intersect is not None:
        result.append((intersect, lastID, sum_score / cnt))

    widths = {}
    heights = {}
    for item in anno["images"]:
        widths[item["id"]] = item["width"]
        heights[item["id"]] = item["height"]

    result = [(bbox, imgID, score) for bbox, imgID, score in result
              if (0 <= bbox[0] <= widths[imgID] and 0 <= bbox[1] <= heights[imgID] and
                  8000 <= bbox[2] * bbox[3] <= 22500)]

    res = []
    for i, d in enumerate(result):
        bbox, imgID, score = d
        res.append(
            {
                "image_id": imgID,
                "category_id": 1,
                "bbox": bbox,
                "score": score
            }
        )
    with open(params.output_path, "w") as file:
        json.dump(res, file, indent=4)


def calcAP50(params):
    cocoGt = COCO(params.anno_path)
    cocoDt = cocoGt.loadRes(params.output_path)
    imgIds = sorted(cocoGt.getImgIds())
    # imgId = imgIds[np.random.randint(100)]
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # with open(params.output_path, "r") as file:
    #     predict = json.load(file)
    # with open(params.anno_path, "r") as file:
    #     target = json.load(file)

    # gt = {}
    # for b in target["annotations"]:
    #     box = b["bbox"]
    #     box[2] += box[0]
    #     box[3] += box[1]
    #     imgID = b["image_id"]
    #     if imgID in gt:
    #         gt[imgID].append(box)
    #     else:
    #         gt[imgID] = [box]

    # pr = {}
    # for b in predict:
    #     box = b["bbox"]
    #     box[2] += box[0]
    #     box[3] += box[1]
    #     score = b["score"]
    #     imgID = b["image_id"]
    #     if imgID in pr:
    #         pr[imgID]["boxes"].append(box)
    #         pr[imgID]["scores"].append(score)
    #     else:
    #         pr[imgID] = {"boxes": [box], "scores": [score]}

    # with open("result/gt.json", "w") as file:
    #     json.dump(gt, file, indent=4)
    # with open("result/pr.json", "w") as file:
    #     json.dump(pr, file, indent=4)
    # print(IOU.get_avg_precision_at_iou(gt, pr))


def doAllTest(params):
    print("Stage 1: Finding spines")
    findSpine(params)
    print("Stage 2: Finding ribs")
    findRibs(params)
    print("Stage 3: Finding factures")
    findFractureClassifier(params)
    print("Stage 4: Post processing")
    postProcess(params)
    print("Stage 5: Evaluating")
    calcAP50(params)


if __name__ == "__main__":
    params = option.read()
    if params.testTarget == "ribTracerDDPG":
        with open("additional_anno/additional_anno_val.json", "r") as file:
            anno = json.load(file)
        showRibTraceTrackDDPG("data/fracture/val_processed/101.png",
                              anno["poly"]["101"][1:], params)
    elif params.testTarget == "spine":
        view.showSpine("101")
    elif params.testTarget == "chest":
        findChest(params)
    elif params.testTarget == "ribs":
        findRibs(params)
        # files = os.listdir(params.data_set)
        # for f in files:
        # view.showRibs(
        # f.split('.')[0],
        # )
    elif params.testTarget == "fracture":
        findFractureClassifier(params)
    elif params.testTarget == "fractureYolo":
        findFractureYolo(params)
    elif params.testTarget == "fractureChest":
        findFractureChestDivide(params)
    elif params.testTarget == "postProcess":
        postProcess(params)
    elif params.testTarget == "AP50":
        calcAP50(params)
    elif params.testTarget == "all":
        doAllTest(params)
    elif params.testTarget == "result":
        # view.showRibs(
        #     8,
        #     params.anno_path,
        #     params.output_path
        # )
        files = os.listdir(params.data_set)
        for f in files:
            view.showRibs(
                f.split('.')[0],
                params.anno_path,
                params.output_path
            )
