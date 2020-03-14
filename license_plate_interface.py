#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      :license_plate_interface.py
@Date      :2020/03/13 17:05:04
@Author    :mrwang
@Version   :1.0
'''

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
from imutils import paths
import random, copy
# sys.path.append("data_preprocessing/license_regression")
# import engine
sys.path.insert(0, '.')
sys.path.append("landmark/license_regression")
import engine
# sys.path.append("/home/mario/Projects/SSD/SSD_mobilenetv2/classifier")
# from network import Net
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd_mobilenetv2_fpn import build_ssd   # train with mobilenet backbone
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = build_ssd('test', 300, 2)    # initialize SSD
net.load_state_dict(torch.load('weights_zhongdong/ssd_mobilenetv2_fpn_20200305_addmiss/mobilenetv2_final.pth'))  # test mobilenet backbone
net.eval()
from clsnetwork import Net

# image = cv2.imread('./data/example.jpg', cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
# %matplotlib inline
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
# here we specify year (07 or 12) and dataset ('test', 'val', 'train') 

color_dict = {'0': 'yellow_card', '1': 'white_card', '2': 'red_card', '3': 'green_card'}
lan_dict = {'0': 'Double_column', '1': 'Single_column'}

'''
# 类Detect_license_plate实例化一个对象RM_DLP
RM_DLP = Detect_license_plate('ssd_mobilenetv2_fpn') 
# 调用类中的 detect()方法
RM_DLP.detect(image)
image = cv2.imread(imgpath)
'''

class Detect_license_plate(object):
    def __init__(self, method):
        self.detect_method = method
    
    ## 车牌的颜色和单双栏分类预测
    def classifier(self, image):
        color_lan_str = ''
        with torch.no_grad():
            color_lan_img = cv2.resize(image, (48, 48), interpolation=cv2.INTER_LINEAR)
            color_lan_img = color_lan_img[:,:,::-1]
            color_lan_img = np.asarray(color_lan_img, 'float32')
            color_lan_img = color_lan_img.transpose((2, 0, 1))
            color_lan_img = (color_lan_img - 127.5) * 0.0078125
            color_lan_img = torch.FloatTensor(color_lan_img).unsqueeze(0)
            color_lan_img = torch.autograd.Variable(color_lan_img).cuda()

            color_model.eval()
            lan_model.eval()
            color_output = color_model(color_lan_img)
            color_pred = color_output.max(1, keepdim=True)[1]
            lan_output = lan_model(color_lan_img)

            lan_pred = lan_output.max(1, keepdim=True)[1]
            color_pred = str(color_pred.cpu().numpy()[0][0])
            lan_pred = str(lan_pred.cpu().numpy()[0][0])

            color_info = color_dict[color_pred]
            lan_info = lan_dict[lan_pred]
        # color_lan_str = color_info + ',' + lan_info

        return color_info, lan_info

    ## 车牌的landmark点回归预测
    def regress_landmark(self, image):
        resized_im = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
        crop_path = 'crop.jpg'
        cv2.imwrite(crop_path, resized_im)

        ## load landmark model
        lmark_model_path = "landmark/license_regression/models/model_rgb/multi_epoch_46_landmark.pkl"

        train_rgb = True
        inputSize=[]

        if train_rgb:
            inputSize=[128, 128, 3]
            imgChannel = 3
        else:
            inputSize=[128, 128, 1]
            imgChannel = 1
        
        eng = engine.TestEngineImg(modelPath=lmark_model_path, inputSize=inputSize)
        eyePoints = eng(crop_path, inputSize)
        repoints = eyePoints.reshape((4,2))

        return repoints

    ## 将车牌的预测车牌坐标点还原到原图上
    def restore(self, repoints, new_nxy):
        xcoord = []
        ycoord = []
        xycoords = []
        nx1, ny1, nx2, ny2 = new_nxy[0], new_nxy[1], new_nxy[2], new_nxy[3]

        nw = nx2 - nx1
        nh = ny2 - ny1
        for j in range(repoints.shape[0]):
            epoint_x = (float(repoints[j][0]) * nw + nx1)
            epoint_y = (float(repoints[j][1]) * nh + ny1)
            xcoord.append(epoint_x) 
            ycoord.append(epoint_y)
            coords = [epoint_x, epoint_y]
            xycoords.append(coords)

        return xcoord, ycoord, xycoords

    def detect(self, image):
        height, width, channel = image.shape
        img_ori = copy.deepcopy(image)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        # print('x size:', x.shape)
        
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
        # print('xx size:', xx.size())

        if torch.cuda.is_available():
            xx = xx.to(device)
        y = net(xx)
        from data import VOC_CLASSES as labels
        detections = y.data
        # scale each detection back up to the image64
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

        for i in range(detections.size(1)):
            j = 0
            # print('score:', detections[0,i,j,0])
            while detections[0,i,j,0] >= 0.45:
                score = detections[0,i,j,0]
                label_name = labels[i-1]
                if(score>=0.45):
                    pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                    coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                    
                    w = int(pt[2]) - int(pt[0])
                    h = int(pt[3]) - int(pt[1])

                    # fix delta x and y
                    delta_x1 = int(w * 0.3)
                    delta_y1 = int(h * 0.3)
                    delta_x2 = int(w * 0.3)
                    delta_y2 = int(h * 0.3)

                    nx1 = max(int(pt[0]) - delta_x1,0)
                    ny1 = max(int(pt[1]) - delta_y1,0)
                    nx2 = min(int(pt[2]) + delta_x2, width)
                    ny2 = min(int(pt[3]) + delta_y2, height)

                    new_nxy = [nx1, ny1, nx2, ny2]

                    nw = nx2 - nx1
                    nh = ny2 - ny1

                    crop_img = img_ori[int(ny1): int(ny2), int(nx1): int(nx2), :]

                    ## 车牌的颜色和单双栏分类预测
                    color_info, lan_info = self.classifier(crop_img)

                    ## 车牌的landmark点回归预测
                    repoints = self.regress_landmark(crop_img)

                    ## 将车牌的landmark点还原到原图上
                    xcoord, ycoord, xycoords = self.restore(repoints, new_nxy)
                    # print('xycoords:', xycoords)
                    
                    ## 将回归的点坐标进行显示
                    for k in range(len(xcoord)):
                        epoint_x = xcoord[k]
                        epoint_y = ycoord[k]
                        cv2.circle(image,(int(epoint_x),int(epoint_y)),5,(255,0,0),-1)

                    xmin = min(xcoord)
                    ymin = min(ycoord)
                    xmax = max(xcoord)
                    ymax = max(ycoord)

                    rect = [xmin, ymin, xmax, ymax]

                    color_lan_str = color_info + ', ' + lan_info
                    cv2.putText(image,str(np.round(score.cpu().numpy(),3)),(int(xmin),int(ymin)),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
                    cv2.putText(image, color_lan_str, (int(xmin),int(ymin)-25), cv2.FONT_HERSHEY_TRIPLEX, 1, color=(255,0,0))
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2) 

                    # savepath = os.path.join(lmark_dir,imagename)
                    # print('savepath:',savepath)
                    # cv2.imwrite(savepath,image) 

                j+=1
            # cv2.imwrite(save_imgpath,image)
        image = cv2.resize(image, (1280, 960))
        cv2.imshow('image', image)
        cv2.waitKey(0)
        return rect, repoints, xycoords, color_info, lan_info

# yellow_card:0
# white_card:1
# red_card:2
# green_card:3
# Double_column:0
# Single_column:1

if __name__ == "__main__":
    
    color_model_path = 'classifier/colormodel/color_epoch29.pth'
    lan_model_path = 'classifier/lanmodel/lan_epoch29.pth'
    color_model = (torch.load(color_model_path))
    lan_model = (torch.load(lan_model_path))
    color_model = color_model.to(device)
    lan_model = lan_model.to(device)
    image_dir = '/media/mario/新加卷/DataSets/videosrc/zhongdong/test'
    img_paths = [el for el in paths.list_images(image_dir)]
    for imgpath in img_paths:
        print(imgpath)
        image = cv2.imread(imgpath)

        RM_DLP = Detect_license_plate('ssd_mobilenetv2_fpn')
        rect, repoints, xycoords, color_info, lan_info = RM_DLP.detect(image)
        print('rect, repoints, xycoords, color_info, lan_info:', rect, repoints, xycoords, color_info, lan_info)
'''
## 输出信息如下：
rect:
[1123.3803560137749, 1212.7393924593925, 1282.9949952363968, 1285.6960911750793] 
repoints:
[[0.1918233  0.24175134]
 [0.7924696  0.21564026]
 [0.8380364  0.7575447 ]
 [0.23366669 0.80400074]] 
xycoords:
 [[1123.3803560137749, 1215.9771665334702], 
 [1271.7399963140488, 1212.7393924593925], 
 [1282.9949952363968, 1279.935542345047], 
 [1133.7156719863415, 1285.6960911750793]] 
 color_info, lan_info:
 red_card Double_column
'''
