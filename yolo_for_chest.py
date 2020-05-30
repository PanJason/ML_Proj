import torchvision.models as tvmodel
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import option
import json
import cv2
import os
num_classes = 1  # 0不是chest,1是chest
B = 2  # 每个cell产生的bbox数量
picture_size = 448  # 输入图片的大小
params = option.read()  # 命令行参数


class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        # 调用torchvision里的resnet34预训练模型
        resnet = tvmodel.resnet34(pretrained=True)
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(
            *list(resnet.children())[:-2])  # 去除resnet的最后两层
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 7*7*(5*B+num_classes)),
            nn.Sigmoid()  # 将输出全部映射到(0,1)之间
        )

    def forward(self, input):
        input = self.resnet(input)
        input = self.Conv_layers(input)
        input = input.view(input.size()[0], -1)
        input = self.Conn_layers(input)
        return input.reshape(-1, 7, 7, 5*B+num_classes)  # reshape一下输出数据


class yoloLoss(nn.Module):
    def __init__(self, l_coord=5, l_noobj=0.5):
        super(yoloLoss, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        xlt = torch.max(
            box1[:, 0].unsqueeze(1).expand(N, M),
            box2[:, 0].unsqueeze(0).expand(N, M),
        )
        ylt = torch.max(
            box1[:, 1].unsqueeze(1).expand(N, M),
            box2[:, 1].unsqueeze(0).expand(N, M),
        )
        xrb = torch.min(
            box1[:, 2].unsqueeze(1).expand(N, M),
            box2[:, 2].unsqueeze(0).expand(N, M),
        )
        yrb = torch.min(
            box1[:, 3].unsqueeze(1).expand(N, M),
            box2[:, 3].unsqueeze(0).expand(N, M),
        )
        inter = ((xrb-xlt)*(yrb-ylt)).float()

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(
            inter).float()  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(
            inter).float()  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+numclasses) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,Bx5+numclasses)
        '''
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:, :, :, 4] > 0
        noo_mask = target_tensor[:, :, :, 4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_target = target_tensor[coo_mask].view(-1, B*5+num_classes)
        box_target = coo_target[:, :B*5].contiguous().view(-1, 5)
        class_target = coo_target[:, B*5:]

        coo_pred = pred_tensor[coo_mask].view(-1, B*5+num_classes)
        box_pred = coo_pred[:, :B*5].contiguous().view(-1, 5)
        class_pred = coo_pred[:, B*5:].contiguous().view(-1, num_classes)

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1, B*5+num_classes)
        noo_target = target_tensor[noo_mask].view(-1, B*5+num_classes)
        noo_pred_mask = torch.zeros_like(noo_pred).byte()
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum')
        # compute contain obj loss
        coo_response_mask = torch.zeros_like(box_target).byte()
        coo_not_response_mask = torch.zeros_like(box_target).byte()
        box_target_iou = torch.zeros_like(box_target).float()
        for i in range(0, box_target.size()[0], 2):
            box1 = box_pred[i:i+2]
            box1_xyxy = torch.zeros_like(box1).float()
            box1_xyxy[:, :2] = box1[:, :2]-0.5*box1[:, 2:4]  # /14?
            box1_xyxy[:, 2:4] = box1[:, :2]+0.5*box1[:, 2:4]
            box2 = box_target[i].unsqueeze(0)
            box2_xyxy = torch.zeros_like(box2).float()
            box2_xyxy[:, :2] = box2[:, :2] - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])
            max_iou, max_index = iou.max(0)
            coo_response_mask[i + max_index] = 1
            coo_not_response_mask[i + 1 - int(max_index), :] = 1
            box_target_iou[i+int(max_index), 4] = max_iou
        # 1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(
            box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum')
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum') + \
            F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]+1e-6), torch.sqrt(
                box_target_response[:, 2:4]+1e-6), reduction='sum')
        # 2.not response loss
        #box_pred_not_response = box_pred[coo_not_response_mask]
        #box_target_not_response = box_target[coo_not_response_mask]
        #box_target_not_response[:, 4] = 0
        #not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], reduction='sum')

        # 3.class loss
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        return (self.l_coord * loc_loss + contain_loss + self.l_noobj * nooobj_loss + class_loss) / N


class chest_dataset(torch.utils.data.Dataset):
    def __init__(self, mode=0):
        """
                mode: 0训练，1验证，2测试
                is_aug:  是否进行数据增广
                """
        self.mode = mode
        if mode == 0:
            self.img_path = 'data/fracture/train'
            self.annotation_path = 'additional_anno/additional_anno_train.json'
        if mode == 1:
            self.img_path = 'data/fracture/val'
            self.annotation_path = 'additional_anno/additional_anno_val.json'
        with open(self.annotation_path, "r") as file:
            anno = json.load(file)
        self.bboxes = anno['bbox']
        self.map = [i for i in self.bboxes.keys()]

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, item):
        img = cv2.imread(self.img_path + '/' + self.map[item] + ".png")
        img = cv2.resize(img, (picture_size, picture_size))
        bbox = np.array(self.bboxes[self.map[item]]) * 448.0 / 3056

        rotate_time = random.randint(0, 3)
        for i in range(rotate_time):
            img, bbox = rotate_90(img, bbox)

        scale = 0.5 + random.random() * 0.5
        img, bbox = warp(img, bbox, scale)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        target = torch.zeros(7, 7, num_classes + B * 5)
        x = bbox[0][0] * 1.0 / 448
        y = bbox[0][1] * 1.0 / 448
        w = bbox[1][0] * 1.0 / 448
        h = bbox[1][1] * 1.0 / 448
        gridsize = 1.0 / 7
        center_x = x + 0.5 * w
        center_y = y + 0.5 * h
        gridx = int(center_x / gridsize)
        gridy = int(center_y / gridsize)
        px = center_x / gridsize - gridx
        py = center_y / gridsize - gridy
        target[gridx, gridy, 0], target[gridx, gridy, 5] = px, px
        target[gridx, gridy, 1], target[gridx, gridy, 6] = py, py
        target[gridx, gridy, 2], target[gridx, gridy, 7] = w, w
        target[gridx, gridy, 3], target[gridx, gridy, 8] = h, h
        target[gridx, gridy, 4], target[gridx, gridy, 9] = 1, 1
        target[gridx, gridy, 10] = 1
        return img, target


def rotate_90(img, bbox):
    # mat rotate 1 center 2 angle 3 缩放系数
    matRotate = cv2.getRotationMatrix2D((224, 224), 90, 1)
    dst = cv2.warpAffine(img, matRotate, (448, 448))
    ox, oy = bbox[0]
    ow, oh = bbox[1]
    w, h = oh, ow
    x = oy
    y = 448 - ox - ow
    return dst, np.array([[x, y], [w, h]])


def warp(img, bbox, scale):
    new_size = int(448 * scale)
    src = cv2.resize(img, (new_size, new_size))
    dst = np.zeros((448, 448, 3))
    dst[0:new_size, 0:new_size, :] = src[0:new_size, 0:new_size, :]
    bbox1 = bbox * scale
    return dst, bbox1


def train():
    epoch = 500
    batchsize = 5
    lr = 0.001

    train_data = chest_dataset()
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batchsize, shuffle=True)
    model = torch.load("./saved_model/YOLOv1_epoch" + str(250) + ".pkl")
    # model.children()里是按模块(Sequential)提取的子模块，而不是具体到每个层，具体可以参见pytorch帮助文档
    # 冻结resnet34特征提取层，特征提取层不参与参数更新
    for layer in model.children():
        layer.requires_grad = False
        break

    criterion = yoloLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 100, 200, 300, 400],
                                                     gamma=0.5)
    for e in range(epoch):
        model.train()
        scheduler.step()
        if e < 250:
            continue
        yl = torch.Tensor([0])
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.cuda()
            labels = labels.float().cuda()
            pred = model(inputs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch %d/%d| Step %d/%d| Loss: %.2f" %
                  (e, epoch, i, len(train_data) // batchsize, loss))
            yl = yl + loss
        if (e + 1) % 50 == 0:
            torch.save(model, "./saved_model/YOLOv1_epoch" +
                       str(e + 1) + ".pkl")
        with open('log.txt', 'a+') as file_object:
            file_object.write("epoch:%d loss:%f\n" % (e, yl / len(train_data)))


def generate_bboxes(pred):
    # pred: tensor batchsize*7*7*12
    # bboxes:tensor batchsize *4 x1 y1 x2 y2
    N = pred.size()[0]
    bboxes = torch.zeros(N, 4).float()
    pred_bboxes = pred[:, :, :, :10].contiguous().view(-1, 5)
    for i in range(N):
        a = pred_bboxes[i*98:(i+1)*98, 4]
        b, c = torch.max(a, 0)
        px, py, w, h = pred_bboxes[i*98+c, 0:4]
        c = int(c)//2
        u = c//7
        v = c % 7
        x = (u+px)*1.0/7
        y = (v+py)*1.0/7
        x1 = min(max(x-0.5*w, 0), 1)*3056
        x2 = min(max(x+0.5*w, 0), 1)*3056
        y1 = min(max(y-0.5*h, 0), 1)*3056
        y2 = min(max(y+0.5*h, 0), 1)*3056
        bboxes[i, 0], bboxes[i, 1], bboxes[i, 2], bboxes[i, 3] = x1, y1, x2, y2
    return bboxes


def test_chest(params):
    model = torch.load(os.path.join(params.model_path, 'yolo_for_chest.pkl'))
    mydevice = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(mydevice)
    files = os.listdir(params.processed)
    result = {}
    for i, file in enumerate(files):
        file_id = file.split('.')[0]
        print(f"testing on image file ({i + 1}/{len(files)}): {file}")
        img = cv2.imread(os.path.join(params.processed, file))
        img = cv2.resize(img, (448, 448))
        img1 = img.transpose(2, 0, 1)
        img1 = torch.from_numpy(img1).unsqueeze(0).float().cuda()
        pred = model(img1)
        bboxes = generate_bboxes(pred)
        x1, y1, x2, y2 = bboxes[0, :]
        w = x2-x1
        h = y2-y1
        bbox = [float(x1), float(y1), float(w), float(h)]
        result[i] = {'bbox': bbox, 'id': i, 'image_id': file_id}
    with open(os.path.join(params.median, "chest.json"), "w") as file:
        json.dump(result, file, indent=4)
    return result


if __name__ == '__main__':
    train()
