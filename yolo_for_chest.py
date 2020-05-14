import torchvision.models as tvmodel
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import option
import json
import cv2
num_classes=2#0不是chest,1是chest
B=2#每个cell产生的bbox数量
picture_size=448#输入图片的大小
params = option.read()#命令行参数
class Yolo(nn.Module):
    def __init__(self):
        super(Yolo,self).__init__()
        resnet = tvmodel.resnet34(pretrained=True)  # 调用torchvision里的resnet34预训练模型
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两层
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel,1024,3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024,1024,3,stride=2,padding=1),
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
            nn.Linear(7*7*1024,4096),
            nn.LeakyReLU(),
            nn.Linear(4096,7*7*(5*B+num_classes)),
            nn.Sigmoid()  # 将输出全部映射到(0,1)之间
        )

    def forward(self, input):
        input = self.resnet(input)
        input = self.Conv_layers(input)
        input = input.view(input.size()[0],-1)
        input = self.Conn_layers(input)
        return input.reshape(-1,7,7,5*B+num_classes)  # reshape一下输出数据

class yoloLoss(nn.Module):
    def __init__(self,l_coord=5,l_noobj=0.5):
        super(yoloLoss,self).__init__()
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
        area1 = area1.unsqueeze(1).expand_as(inter).float()  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter).float()  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+numclasses) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,Bx5+numclasses)
        '''
        N = pred_tensor.size()[0]
        coo_mask=target_tensor[:,:,:,4]>0
        noo_mask = target_tensor[:, :, :, 4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_target = target_tensor[coo_mask].view(-1, B*5+num_classes)
        box_target = coo_target[:, :B*5].contiguous().view(-1, 5)
        class_target = coo_target[:, B*5:]

        coo_pred=pred_tensor[coo_mask].view(-1,B*5+num_classes)
        box_pred=coo_pred[:,:B*5].contiguous().view(-1,5)
        class_pred=coo_pred[:,B*5:].contiguous().view(-1,num_classes)

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1, B*5+num_classes)
        noo_target = target_tensor[noo_mask].view(-1, B*5+num_classes)
        noo_pred_mask=torch.zeros_like(noo_pred).byte()
        noo_pred_mask[:,4]=1
        noo_pred_mask[:,9]=1
        noo_pred_c=noo_pred[noo_pred_mask]
        noo_target_c=noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum')
        # compute contain obj loss
        coo_response_mask=torch.zeros_like(box_target).byte()
        coo_not_response_mask=torch.zeros_like(box_target).byte()
        box_target_iou=torch.zeros_like(box_target).float()
        for i in range(0,box_target.size()[0],2):
            box1=box_pred[i:i+2]
            box1_xyxy=torch.zeros_like(box1).float()
            box1_xyxy[:,:2]=box1[:,:2]-0.5*box1[:,2:4]#/14?
            box1_xyxy[:,2:4]=box1[:,:2]+0.5*box1[:,2:4]
            box2=box_target[i].unsqueeze(0)
            box2_xyxy=torch.zeros_like(box2).float()
            box2_xyxy[:, :2] = box2[:, :2] - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] + 0.5 * box2[:, 2:4]
            iou=self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4])
            max_iou,max_index=iou.max(0)
            coo_response_mask[i + max_index] = 1
            coo_not_response_mask[i + 1 - int(max_index),:] = 1
            box_target_iou[i+int(max_index),4]=max_iou
        #1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum')
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum') + \
                   F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]+1e-6), torch.sqrt(box_target_response[:, 2:4]+1e-6), reduction='sum')
        # 2.not response loss
        #box_pred_not_response = box_pred[coo_not_response_mask]
        #box_target_not_response = box_target[coo_not_response_mask]
        #box_target_not_response[:, 4] = 0
        #not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], reduction='sum')

        # 3.class loss
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        return (self.l_coord * loc_loss + contain_loss + self.l_noobj * nooobj_loss + class_loss) / N

class chest_dataset(torch.utils.data.Dataset):
    def __init__(self,mode=0):
        """
                mode: 0训练，1验证，2测试
                is_aug:  是否进行数据增广
                """
        self.mode=mode
        if mode==0:
            self.img_path=params.data_set
            self.annotation_path =params.addi_path
        if mode==1:
            self.img_path=params.val_data_set
            self.annotation_path=params.val_addi_path
        with open(self.annotation_path, "r") as file:
            anno = json.load(file)
        self.bboxes=anno['bbox']
        self.map=[i for i in self.bboxes.keys()]
    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, item):
        img=cv2.imread(self.img_path+'/'+self.map[item]+".png")
        img=cv2.resize(img,(picture_size,picture_size))
        img = img.transpose(2, 0, 1)
        img=torch.from_numpy(img).float()
        target=torch.zeros(7,7,num_classes+B*5)
        x=self.bboxes[self.map[item]][0][0]*1.0/3056
        y=self.bboxes[self.map[item]][0][1]*1.0/3056
        w=self.bboxes[self.map[item]][1][0]*1.0/3056
        h=self.bboxes[self.map[item]][1][1]*1.0/3056
        gridsize=1.0/7
        for i in range(7):
            for j in range(7):
                if (x<=i*gridsize<=x+w or x<=(i+1)*gridsize<=x+w\
                    and y<=j*gridsize<=y+h or y<=(j+1)*gridsize<=y+h):
                    target[i,j,0],target[i,j,5]=x+0.5*w,x+0.5*w
                    target[i,j,1],target[i,j,6]=y+0.5*h,y+0.5*h
                    target[i,j,2],target[i,j,7]=w,w
                    target[i,j,3],target[i,j,8]=h,h
                    target[i,j,4],target[i,j,9]=1.0,1.0
                    target[i,j,11]=1.0
        return img,target

def train():
    epoch = 50
    batchsize = 5
    lr = 0.001

    train_data = chest_dataset()
    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=batchsize,shuffle=True)
    model = Yolo()
    if params.useGPU:
        model = model.cuda()
    # model.children()里是按模块(Sequential)提取的子模块，而不是具体到每个层，具体可以参见pytorch帮助文档
    # 冻结resnet34特征提取层，特征提取层不参与参数更新
    for layer in model.children():
        layer.requires_grad = False
        break
    criterion = yoloLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
    for e in range(epoch):
        model.train()
        yl = torch.Tensor([0])
        for i,(inputs,labels) in enumerate(train_dataloader):
            inputs = inputs
            labels = labels.float()
            pred = model(inputs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch %d/%d| Step %d/%d| Loss: %.2f"%(e,epoch,i,len(train_data)//batchsize,loss))
            yl = yl + loss
        if (e+1)%10==0:
            torch.save(model,"./saved_model/YOLOv1_epoch"+str(e+1)+".pkl")
        with open('log.txt', 'a+') as file_object:
            file_object.write("epoch:%d loss:%f\n" % (e, yl/len(train_data)))
if __name__=='__main__':
    train()