# Do data processing in this file
import math
import json
import os
import random
import itertools

from PIL import Image
import PIL
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms.functional as TransFunc

import option


def dataprocess(input_img):
    padding_size = (3056, 3056, 3)
    output_img = np.zeros(padding_size)
    w = min(padding_size[0], input_img.shape[0])
    h = min(padding_size[1], input_img.shape[1])
    output_img[0:w, 0:h] = input_img[0:w, 0:h]

    huidu = 160
    output_img = output_img*160/np.mean(input_img)
    return output_img


def getAnno(imgId, anno):
    '''Get all annotation of specified image from a parsed json file,
returns a list of bounding boxes.
    '''
    return [i["bbox"] for i in anno["annotations"] if i["image_id"] == imgId]


def getImageFiles(dataset_path):
    'Get all image files inside a dataset'
    files = os.listdir(dataset_path)
    return files


class RibTraceDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, anno_path, params, augment=False):
        super(RibTraceDataset, self).__init__()
        self.params = params
        self.dataset = dataset
        self.sampleRate = params.tracerSampleRate
        self.augment = augment

        self.anno_path = anno_path
        with open(self.anno_path, "r") as file:
            self.anno = json.load(file)["poly"]

        self.annoted_id = list(self.anno.keys())
        self.annoted_files = [
            (self.dataset + "/" + s + ".png")
            for s in self.annoted_id
        ]
        self.file_cnt = len(self.annoted_files)

        self.regionSize = params.regionSize
        self.width = params.regionSize
        self.height = params.regionSize

    def gen_data(self, start, end):
        for i in range(start, end):
            image = Image.open(self.annoted_files[i])
            image = torchvision.transforms.Pad(int(self.regionSize/2))(image)
            point_chunks = []
            for poly in self.anno[self.annoted_id[i]][1:]:
                poly = np.array(poly)
                point_chunks.append(self.samplePoints(poly))
            for point in itertools.chain(*point_chunks):
                yield self.getRegion(image, point)

    def samplePoints(self, poly):
        points = []
        for i in range(0, len(poly)-1):
            length = np.linalg.norm(poly[i] - poly[i+1])
            sample_cnt = int(self.sampleRate * length)
            samples = [(
                (poly[i+1] * (j/sample_cnt) + poly[i] * (1-(j/sample_cnt))),
                poly[i],
                poly[i+1]
            )
                for j in range(sample_cnt)]
            points.append(samples)
        return itertools.chain(*points)

    def getRegion(self, image, points):
        center, A, B = points
        result = np.zeros((self.width, self.height))

        def shift(x):
            y = random.gauss(x, self.params.regionShiftSigma)
            y = np.clip(y, x - self.regionSize / 2, x + self.regionSize / 2)
            y = int(round(y))
            return y
        if self.augment:
            C = np.array([shift(center[0]), shift(center[1])])
        else:
            C = center

        region = TransFunc.crop(
            image, C[1], C[0], self.regionSize, self.regionSize)

        P = (B-center) / np.linalg.norm(B-center) * \
            min(self.regionSize / 2, np.linalg.norm(B-center))
        target = P + C - center
        target /= self.regionSize * math.sqrt(2) / 2

        if self.augment:
            angle = random.uniform(-180, 180)
            region = TransFunc.rotate(region, angle, resample=Image.BILINEAR)
            angle = -angle / 180.0 * math.pi
            target = [math.cos(angle) * target[0] - math.sin(angle) * target[1],
                      math.sin(angle) * target[0] + math.cos(angle) * target[1]]
        if target[0] < 0.0:
            target = [-target[0], -target[1]]
        target[1] = target[1] / 2.0 + 0.5

        region = torchvision.transforms.ToTensor()(region)
        target = torch.tensor(target, dtype=torch.float32)
        # if self.params.useGPU:
        #     region = region.cuda()
        #     target = target.cuda()
        return (region, target)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return self.gen_data(0, self.file_cnt)
        else:  # in a worker process
            per_worker = int(
                math.ceil(self.file_cnt / float(worker_info.num_workers)))
            worker_id = worker_info.id
            return self.gen_data(worker_id * per_worker, min((worker_id+1) * per_worker, self.file_cnt))


class RibTraceDDPGDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, anno_path, params, augment=False):
        super(RibTraceDDPGDataset, self).__init__()
        self.params = params
        self.dataset = dataset
        self.sampleRate = params.tracerSampleRate
        self.augment = augment

        self.anno_path = anno_path
        with open(self.anno_path, "r") as file:
            self.anno = json.load(file)["poly"]

        self.annoted_id = list(self.anno.keys())
        if augment:
            random.shuffle(self.annoted_id)
        self.annoted_files = [
            (self.dataset + "/" + s + ".png")
            for s in self.annoted_id
        ]
        self.file_cnt = len(self.annoted_files)

        self.regionSize = params.regionSize
        self.width = params.regionSize
        self.height = params.regionSize

    def gen_data(self, start, end):
        for i in range(start, end):
            image = Image.open(self.annoted_files[i])

            C = np.array([self.params.imageSize/2, self.params.imageSize/2])
            for poly in self.anno[self.annoted_id[i]][1:]:
                poly = np.array(poly)
                if self.augment:
                    angle = random.uniform(-180, 180)
                    img = TransFunc.rotate(
                        image, angle, resample=Image.BILINEAR)
                    angle = -angle / 180.0 * math.pi
                    for i in range(len(poly)):
                        t = poly[i] - C
                        t = np.array([math.cos(angle) * t[0] - math.sin(angle) * t[1],
                                      math.sin(angle) * t[0] + math.cos(angle) * t[1]])
                        poly[i] = C + t
                else:
                    img = image

                img = torchvision.transforms.Pad(int(self.regionSize/2))(img)
                # img = torchvision.transforms.ToTensor()(img)
                yield img, poly

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return self.gen_data(0, self.file_cnt)
        else:  # in a worker process
            per_worker = int(
                math.ceil(self.file_cnt / float(worker_info.num_workers)))
            worker_id = worker_info.id
            return self.gen_data(worker_id * per_worker, min((worker_id+1) * per_worker, self.file_cnt))


class VAEDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, params, augment=False):
        super(VAEDataset, self).__init__()
        self.params = params
        self.dataset = dataset
        self.augment = augment

        self.files = os.listdir(params.data_set)
        self.file_cnt = len(self.files)

        self.regionSize = params.regionSize
        self.width = params.regionSize
        self.height = params.regionSize

    def gen_data(self, start, end):
        for i in range(start, end):
            image = Image.open(os.path.join(
                self.params.data_set, self.files[i]))
            image = torchvision.transforms.Pad(int(self.regionSize/2))(image)
            for k in range(self.params.VAESamples):
                pos = (random.randint(0, self.params.imageSize),
                       random.randint(0, self.params.imageSize))
                img = torchvision.transforms.functional.crop(
                    image, pos[0], pos[1], self.regionSize, self.regionSize)
                img = torchvision.transforms.ToTensor()(img)
                yield img

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return self.gen_data(0, self.file_cnt)
        else:  # in a worker process
            per_worker = int(
                math.ceil(self.file_cnt / float(worker_info.num_workers)))
            worker_id = worker_info.id
            return self.gen_data(worker_id * per_worker, min((worker_id+1) * per_worker, self.file_cnt))


class ClassifierTestDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, anno, params):
        super(ClassifierTestDataset, self).__init__()
        self.params = params
        self.sampleRate = params.detectRate
        self.anno = anno

        self.files = os.listdir(dataset)
        self.file_cnt = len(self.files)
        self.ids = [i.split('.')[0] for i in self.files]
        self.files = [os.path.join(dataset, i) for i in self.files]

        self.regionSize = params.detectRegionSize

    def gen_data(self, start, end):
        for i in range(start, end):
            image = Image.open(self.files[i])
            image = image.convert("RGB")
            image = torchvision.transforms.Pad(int(self.regionSize/2))(image)
            point_chunks = []
            for poly in self.anno[self.ids[i]]:
                poly = np.array(poly)
                point_chunks.append(self.samplePoints(poly, self.ids[i]))
            for point in itertools.chain(*point_chunks):
                yield self.getRegion(image, point)

    def samplePoints(self, poly, imgID):
        points = []
        for i in range(0, len(poly)-1):
            length = np.linalg.norm(poly[i] - poly[i+1])
            sample_cnt = int(length / self.sampleRate)
            samples = [(
                (poly[i+1] * (j/sample_cnt) + poly[i] * (1-(j/sample_cnt))),
                imgID
            )
                for j in range(sample_cnt)]
            points.append(samples)
        return itertools.chain(*points)

    def getRegion(self, image, info):
        center, imgID = info
        imgID = int(imgID)
        region = TransFunc.crop(
            image, center[1], center[0], self.regionSize, self.regionSize)

        region = torchvision.transforms.ToTensor()(region)
        center = torch.tensor(center, dtype=float)
        imgID = torch.tensor(imgID, dtype=int)
        return region, center, imgID

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return self.gen_data(0, self.file_cnt)
        else:  # in a worker process
            per_worker = int(
                math.ceil(self.file_cnt / float(worker_info.num_workers)))
            worker_id = worker_info.id
            return self.gen_data(worker_id * per_worker, min((worker_id+1) * per_worker, self.file_cnt))


class ChestDivideDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, anno, params):
        super(ChestDivideDataset, self).__init__()
        self.params = params
        self.sampleRate = params.detectRate

        with open(anno, "r") as file:
            self.anno = json.load(file)

        self.files = os.listdir(dataset)
        self.file_cnt = len(self.files)
        self.ids = [i.split('.')[0] for i in self.files]
        self.files = [os.path.join(dataset, i) for i in self.files]

        self.regionSize = params.detectRegionSize

    def gen_data(self, start, end):
        for i in range(start, end):
            image = Image.open(self.files[i])
            image = image.convert("RGB")
            image = torchvision.transforms.Pad(int(self.regionSize/2))(image)

            box = self.anno[self.ids[i]]
            # print(box)

            wcnt = int((box[2] - box[0]) / self.sampleRate)
            hcnt = int((box[3] - box[1]) / self.sampleRate)

            for u in range(wcnt):
                for v in range(hcnt):
                    yield self.getRegion(image,
                                         (
                                             (
                                                 int(box[0] * (1 - u / wcnt) +
                                                     box[2] * (u / wcnt)),
                                                 int(box[1] * (1 - v / hcnt) +
                                                     box[3] * (v / hcnt))
                                             ),
                                             self.ids[i],
                                         ))

    def getRegion(self, image, info):
        center, imgID = info
        imgID = int(imgID)
        # print(center)
        region = TransFunc.crop(
            image, center[1], center[0], self.regionSize, self.regionSize)

        region = torchvision.transforms.ToTensor()(region)
        center = torch.tensor(center, dtype=float)
        imgID = torch.tensor(imgID, dtype=int)
        return region, center, imgID

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return self.gen_data(0, self.file_cnt)
        else:  # in a worker process
            per_worker = int(
                math.ceil(self.file_cnt / float(worker_info.num_workers)))
            worker_id = worker_info.id
            return self.gen_data(worker_id * per_worker, min((worker_id+1) * per_worker, self.file_cnt))
