from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import torch
import cv2
import numpy as np


def detectFracture(img, cfg='cfg/yolov3-tiny3-1cls.cfg', weights='weights/detector.pt', half=False,
                   augment=True, conf_thres=0.5, iou_thres=0.5, agnostic_nms=False):
    '''
    :param img: a list of images of size (H, W, C), or an image of size (H, W, C)
    :return: a dict describing the bbox in each image
        if the ith image do not have an output of bboxes, i is not in dict
        else dict[i] is the max confidence bbox in ith image, [xmin, ymin, weight, height]
    '''
    imgsz = 416
    for i in range(len(img)):
        img[i] = cv2.resize(img[i], (416, 416))
        img[i] = np.transpose(img[i], (2, 0, 1))

    device_available = '0' if torch.cuda.is_available() else 'cpu'

    device = torch_utils.select_device(device=device_available)

    model = Darknet(cfg, imgsz)

    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    print('load yolo tiny successfully')

    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'
    if half:
        model.half()

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        inf_out = model(img, augment=augment)[0]

        # to float
        if half:
            inf_out = inf_out.float()

        output = non_max_suppression(inf_out, conf_thres, iou_thres,
                               multi_label=False, agnostic=agnostic_nms)

    pred_dict = dict()
    for i, pred in enumerate(output):
        if pred is None:
            continue
        # clip_coords(pred, (imgsz, imgsz))
        box = pred[:, :4].clone() # xyxy
        # scale_coords(img[i].shape[1:], box, imgsz, imgsz)  # to original shape
        box = xyxy2xywh(box)  # xywh
        box[:, :2] -= box[:, 2:] / 2
        # select max confidence
        max_score = 0
        for p, b in zip(pred.tolist(), box.tolist()):
            bbox = [round(x, 3) for x in b]
            score = round(p[4], 5)
            if score > max_score:
                max_score = score
                pred_dict[i] = bbox

    return pred_dict

'''
img = cv2.imread('../detectorData/val/images/0.png')
# img = cv2.resize(img, (416, 416))
# img = np.transpose(img, (2, 0, 1))
detectFracture(img)
'''


