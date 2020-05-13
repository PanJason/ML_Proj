# prepare data for the classifier

import numpy as np
import json
from PIL import Image
import random

sizen = 360
json_train = 'fracture/annotations/anno_train.json'
json_val = 'fracture/annotations/anno_val.json'
image_train = 'fracture/train/'
image_val = 'fracture/val/'
image_train_save = 'data/train/1/'
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
        # im_num = random.randint(1, 3)
        im_num = 1
        for k in range(im_num):
            ranx = random.uniform(max(0, x + sizen - image_size[image_cnt]['width']), min(sizen - width, x))
            rany = random.uniform(max(0, y + sizen - image_size[image_cnt]['height']), min(sizen - height, y))
            region = im.crop((x - ranx, y - rany, x - ranx + sizen, y - rany + sizen))
            region.save(image_save_path+str(cnt)+'.png', 'png')
            cnt += 1

    print(cnt)


def create_nonfracture(json_path, image_path, image_save_path):
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

    for image_id in image_dict:
        width = image_dict[image_id][0]
        height = image_dict[image_id][1]
        im = Image.open(image_path + str(image_id) + '.png', 'r')

        min_x = image_dict[image_id][2][0]
        max_x = image_dict[image_id][2][0] + image_dict[image_id][2][2]
        min_y = image_dict[image_id][2][1]
        max_y = image_dict[image_id][2][1] + image_dict[image_id][2][3]

        for t in image_dict[image_id][3:]:
            x, y, w, h = t
            min_x = min(x, min_x)
            max_x = max(x + w, max_x)
            min_y = min(y, min_y)
            max_y = max(y + h, max_y)
        k = 4
        try_time = 100
        while k and try_time:
            ranx = random.uniform(min_x - 300, max_x + 50)
            rany = random.uniform(min_y - 300, max_y + 50)
            try_time -= 1
            if ranx + sizen > width or rany + sizen > height:
                continue
            success = True
            for t in image_dict[image_id][3:]:
                x, y, w, h = t
                if ranx + sizen <= x or ranx >= x + width or rany + sizen <= y or rany >= y + height:
                    continue
                else:
                    success = False
                    break
            if success:
                region = im.crop((ranx, rany, ranx + sizen, rany + sizen))
                region.save(image_save_path +str(cnt) + '.png', 'png')
                cnt += 1
                print(cnt)
                k -= 1


create_nonfracture(json_val, image_val, image_val_save)



