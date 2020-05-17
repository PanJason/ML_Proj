import numpy as np
import cv2 as cv
import json
import random
import os
import os.path as path

import data
import option

params = option.read()


def scaling(img):
    height, width = img.shape[:2]
    targeth, targetw = 600, 600
    img = cv.resize(img, (targeth, targetw), interpolation=cv.INTER_AREA)
    hscale = targeth / float(height)
    wscale = targetw / float(width)
    return img, hscale, wscale


def drawPoly(img, poly, color, number):
    if len(poly) > 1:
        for i in range(len(poly)-1):
            cv.line(img, poly[i], poly[i+1], color, 2)
    if len(poly) > 0:
        cv.putText(img, str(number), poly[0],
                   cv.FONT_HERSHEY_SIMPLEX, 1, color)


def randomColor():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def showImage(imgID, imgPath=None, annoPath=None, additionalAnno=None):
    'Show one image with annotation'
    if imgPath is None:
        imgPath = params.data_set + "/" + str(imgID) + ".png"
    if annoPath is None:
        annoPath = params.anno_path

    img = cv.imread(imgPath)
    with open(annoPath, "r") as file:
        anno = json.load(file)

    img, hscale, wscale = scaling(img)

    boxes = data.getAnno(imgID, anno)
    for box in boxes:
        left = box[0] * wscale
        top = box[1] * hscale
        right = left + box[2] * wscale
        down = top + box[3] * wscale
        cv.rectangle(img,
                     (int(left), int(top)),
                     (int(right), int(down)),
                     (0, 255, 255),
                     2)

    if additionalAnno is not None:
        with open(additionalAnno, "r") as file:
            anno = json.load(file)
        bbox = anno["bbox"][str(imgID)]
        print(bbox)
        left = bbox[0][0] * wscale
        top = bbox[0][1] * hscale
        right = left + bbox[1][0] * wscale
        down = top + bbox[1][1] * wscale
        cv.rectangle(img,
                     (int(left), int(top)),
                     (int(right), int(down)),
                     (255, 0, 0),
                     1)

        if str(imgID) in anno["poly"]:
            color = (0, 0, 255)
            for i, poly in enumerate(anno["poly"][str(imgID)]):
                p = [(int(i[0]*wscale), int(i[1]*hscale)) for i in poly]
                drawPoly(img, p, color, i+1)
                color = (random.randint(0, 255), random.randint(
                    0, 255), random.randint(0, 255))

    cv.imshow("Display", img)
    cv.waitKey(0)


def boundBoxAnno(imgID):
    '''Make annotation for bound box of chest for one image.
    Returns [[left, top],[width, height]].
    Operations:
    left click to mark a point.
    First the Leftmost topmost point, then the rightmost bottomost point.
    Click the third time to cancel all.
    Press any key to accept the marking and return the box.
    '''
    imgPath = params.data_set + "/" + str(imgID) + ".png"
    img = cv.imread(imgPath)

    cnt = 0

    img, hscale, wscale = scaling(img)
    cv.putText(img, "ID: "+str(imgID), (0, 30),
               cv.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
    cv.imshow("Display", img)
    bbox = np.array([[-1, -1], [-1, -1]])

    def mouseCallback(event, x, y, flags, param):
        nonlocal cnt
        nonlocal img
        if event == cv.EVENT_LBUTTONUP:
            if cnt < 2:
                bbox[cnt] = np.array([int(x), int(y)])
                cnt += 1
            else:
                cnt = 0
                img, _, _ = scaling(cv.imread(imgPath))
                cv.putText(img, "ID: "+str(imgID), (0, 30),
                           cv.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
            if cnt == 2:
                cv.rectangle(img, (bbox[0][0], bbox[0][1]),
                             (bbox[1][0], bbox[1][1]), (255, 0, 0), 1)

    cv.setMouseCallback("Display", mouseCallback)
    while True:
        cv.imshow("Display", img)
        key = cv.waitKey(50)
        if key != -1:
            break

    bbox = np.array(bbox, dtype=float)
    bbox /= [wscale, hscale]
    bbox[1] -= bbox[0]
    return bbox.tolist()


def polyLineAnno(imgID):
    '''Make annotion with polylines.
    Returns a list of polylines [(x1,y1),(x2,y2),...,(xn,yn)].
    Operations:
    Left click to mark a point, consequtive points are linked together.
    The first polyline should mark the outline of chest bones.
    The other polylines should mark each ribs, start from the outline to the spine.
    Ribs should be marked clockwise.
    Press n for next polyline.
    Press f to stop recording and return.
    '''
    imgPath = params.data_set + "/" + str(imgID) + ".png"
    img = cv.imread(imgPath)
    img, hscale, wscale = scaling(img)
    cv.putText(img, "ID: "+str(imgID), (0, 30),
               cv.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
    cv.imshow("Display", img)

    polycnt = 1
    poly = []
    lineColor = (0, 0, 255)
    mask = np.full((img.shape[0], img.shape[1], 1), 255, np.uint8)
    old = cv.copyTo(img, mask)

    def mouseCallback(event, x, y, flags, param):
        nonlocal poly
        nonlocal img
        if event == cv.EVENT_LBUTTONUP:
            poly.append((x, y))
            if len(poly) > 1:
                cv.line(img, poly[-2], poly[-1], lineColor, 2)
            else:
                cv.putText(img, str(polycnt), (x, y),
                           cv.FONT_HERSHEY_SIMPLEX, 1, lineColor)
        if event == cv.EVENT_RBUTTONUP:
            if len(poly) > 0:
                poly = poly[:-1]
                img = cv.copyTo(old, mask)
                drawPoly(img, poly, lineColor, polycnt)

    result = []
    cv.setMouseCallback("Display", mouseCallback)
    while True:
        cv.imshow("Display", img)
        key = cv.waitKey(50)
        if key != -1:
            if (key & 0xFF) in [ord('n'), ord('f')]:
                if len(poly) > 1:
                    result.append((np.array(poly, dtype=float) /
                                   [wscale, hscale]).tolist())
                    polycnt += 1
                lineColor = (random.randint(0, 255), random.randint(
                    0, 255), random.randint(0, 255))
                poly = []
                old = cv.copyTo(img, mask)
            if (key & 0xFF) == ord('f'):
                break
    return result


def makeAllBBoxAnno(save_path):
    'Mark all chest bound boxes of the data set(specified with program arguments --data_set).'
    bboxes = {}
    allFiles = os.listdir(params.data_set)
    for f in allFiles:
        num = int(f.split('.')[0])
        bbox = boundBoxAnno(num)
        bboxes[num] = bbox
        with open(save_path, "w") as saveFile:
            json.dump(bboxes, saveFile, indent=4)


def makeAllPolyAnno(save_path, start_from=None, total=None):
    '''Mark all polylines of the data set(specified with program arguments --data_set).
    Use start_from argument to specify which one to start with.
    (Note that the order of pictures is lexigraphical order rather than number order due to os.listdir)
    Use total to specify how many picture you need to make annotation.
    '''
    polys = {}
    allFiles = os.listdir(params.data_set)
    if total is None:
        total = len(allFiles)
    start = start_from is None
    for f in allFiles:
        num = int(f.split('.')[0])
        print(num)
        if num == start_from:
            start = True
        if start:
            poly = polyLineAnno(num)
            polys[num] = poly
            with open(save_path, "w") as saveFile:
                json.dump(polys, saveFile, indent=4)
            if len(polys) >= total:
                break


if __name__ == "__main__":
    # makeAllPolyAnno("poly_train_139.json", 139, total=1)
    allFiles = os.listdir(params.data_set)
    for f in allFiles:
        num = int(f.split('.')[0])
        print(num)
        showImage(
            num, additionalAnno="data/fracture/annotations/additional_anno_train.json")
    # print(boundBoxAnno(139))
