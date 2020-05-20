# prepare data for the classifier

import numpy as np
import json
from PIL import Image
import random
import os

sizen = 360
json_train = 'fracture/annotations/anno_train.json'
json_val = 'fracture/annotations/anno_val.json'
json_add_train = 'additional_anno/additional_anno_train.json'
json_add_val = 'additional_anno/additional_anno_val.json'
image_train = 'fracture/train/'
image_val = 'fracture/val/'
image_train_save = 'data/train/0/'
image_val_save = 'data/val/0/'


def create_fracture(json_path, image_path, image_save_path):
    with open(json_path) as f:
        instances = json.load(f)

    anno = instances['annotations']
    image_size = instances['images']
    print(len(image_size))

    cnt = 0
    image_cnt = 0

    for i in range(len(anno)):
        x, y, width, height = anno[i]['bbox']
        image_id = anno[i]['image_id']
        while image_id > image_size[image_cnt]['id']:
            image_cnt += 1
            # print(image_cnt)
        while image_id < image_size[image_cnt]['id']:
            image_cnt -= 1
        im = Image.open(image_path+str(image_id)+'.png', 'r')
        im_num = random.randint(1, 3)
        # im_num = 1
        for k in range(im_num):
            ranx = random.uniform(max(0, x + sizen - image_size[image_cnt]['width']), min(sizen - width, x))
            rany = random.uniform(max(0, y + sizen - image_size[image_cnt]['height']), min(sizen - height, y))
            region = im.crop((x - ranx, y - rany, x - ranx + sizen, y - rany + sizen))
            region.save(image_save_path+str(cnt)+'.png', 'png')
            cnt += 1

    print(cnt)


def create_nonfracture(json_path, json_add_path, image_path, image_save_path):
    with open(json_path) as f:
        instances = json.load(f)

    anno = instances['annotations']
    image_size = instances['images']
    print(len(image_size))
    image_dict = {}

    cnt = 0
    image_cnt = 0
    for i in range(len(anno)):
        x, y, width, height = anno[i]['bbox']
        image_id = anno[i]['image_id']
        while image_id > image_size[image_cnt]['id']:
            image_cnt += 1
            # print(image_cnt)
        while image_id < image_size[image_cnt]['id']:
            image_cnt -= 1
        if image_id in image_dict:
            image_dict[image_id].append((x, y, width, height))
        else:
            image_dict[image_id] = [image_size[image_cnt]['width'], image_size[image_cnt]['height']]
            image_dict[image_id].append((x, y, width, height))

    with open(json_add_path) as f:
        instances = json.load(f)

    anno_add = instances['poly']

    for image_id in anno_add:
        if len(anno_add[image_id]) > 1:
            im = Image.open(image_path + str(image_id) + '.png', 'r')
        # print(image_id)
        width = image_dict[int(image_id)][0]
        height = image_dict[int(image_id)][1]

        for i in range(1, len(anno_add[image_id])):
            for k in range(len(anno_add[image_id][i]) - 1):
                x1, y1 = anno_add[image_id][i][k]
                x2, y2 = anno_add[image_id][i][k+1]
                u = random.uniform(0.2, 0.8)
                ranx = u*x1 + (1-u)*x2 - 180
                rany = u*y1 + (1-u)*y2 - 180
                if ranx + sizen > width or rany + sizen > height or ranx < 0 or rany < 0:
                    continue
                success = True
                for t in image_dict[int(image_id)][3:]:
                    x, y, w, h = t
                    if ranx + sizen <= x or ranx >= x + width or rany + sizen <= y or rany >= y + height:
                        continue
                    else:
                        success = False
                        break
                if success:
                    region = im.crop((ranx, rany, ranx + sizen, rany + sizen))
                    region.save(image_save_path + str(cnt) + '.png', 'png')
                    cnt += 1

    print(cnt)


create_nonfracture(json_val, json_add_val, image_val, image_val_save)




