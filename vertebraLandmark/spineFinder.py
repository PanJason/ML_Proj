import torch
import numpy as np
from vertebraLandmark.models import spinal_net
import cv2
from . import decoder
import os
from vertebraLandmark.dataset import BaseDataset
from . import draw_points
import json


def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes,
                 'reg': 2*args.num_classes,
                 'wh': 2*4, }

        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256,
                                         args=args)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(
            K=args.K, conf_thresh=args.conf_thresh)
        self.dataset = {'spinal': BaseDataset}

    def load_model(self, model, resume):
        checkpoint = torch.load(
            resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(
            resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def map_mask_to_image(self, mask, img, color=None):
        if color is None:
            color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)

    def test(self, args):
        self.model = self.load_model(
            self.model, os.path.join(args.model_path, "vertebraLandmark.pth"))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset["spinal"]
        dsets = dataset_module(data_dir=args.processed,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)

        result = {}
        for cnt, data_dict in enumerate(data_loader):
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]
            images = images.to('cuda')
            print('processing {}/{} image ... {}'.format(cnt,
                                                         len(data_loader), img_id))
            with torch.no_grad():
                output = self.model(images)
                hm = output['hm']
                wh = output['wh']
                reg = output['reg']

            torch.cuda.synchronize(self.device)
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)   # 17, 11
            # print('totol pts num is {}'.format(len(pts2)))
            pts0 = pts2.copy()
            pts0[:, :10] *= args.down_ratio
            pts0 = np.asarray(pts0, np.float32)
            wscale = float(args.imageSize) / args.input_w
            hscale = float(args.imageSize) / args.input_h
            pts0[:, 0::2] = pts0[:, 0::2] * wscale
            pts0[:, 1::2] = pts0[:, 1::2] * hscale
            sort_ind = np.argsort(pts0[:, 1])
            pts0 = pts0[sort_ind]

            result[img_id.split('.')[0]] = [
                [
                    [float(p[0]), float(p[1])],
                    [float(p[2]), float(p[3])],
                    [float(p[4]), float(p[5])],
                    [float(p[6]), float(p[7])],
                    [float(p[8]), float(p[9])]
                ]
                for p in pts0
            ]

        with open(os.path.join(args.median, "spine.json"), "w+") as file:
            json.dump(result, file, indent=4)
        return result
