'''
@Author: Jiangtao
@Date: 2019-08-07 10:42:06
@LastEditors: Jiangtao
@LastEditTime: 2019-09-23 15:09:21
@Description: 
'''

import torch
import cv2
import sys
import numpy as np
from torchvision import transforms as tf
import data_process
import util
import random
import time
# import dlib
import os
from collections import OrderedDict


#from network import backbone

import loss as ls

# if sys.platform == "win32":
#     sys.path.append("c:/Users/Streamax-JT/Documents/landmark_regression")
#     sys.path.append("C:/code/YoloV2")

# from FaceDetection import faceDetector

# import file_operation as fo
# import pts_operation as po
#import img_and_pts_aug as ipa

# __all__ = ['TestEngineVideo', 'TestEngineImg']
__all__ = ['TestEngineImg']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detected = False
dets = [[0,0,0,0]]
# Detector = faceDetector()

def draw_points(img,points):

    print('points:', points)
    points *= 128
    
    points = points.reshape((4,2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(points.shape[0]):
        cv2.circle(img,(int(points[i][0]),int(points[i][1])),2,(255,0,0))
        cv2.putText(img,str(i),(int(points[i][0]),int(points[i][1])), font, 0.4, (255, 255, 255), 1)
    return img

class TestEngineImg():
    """ This is a custom engine for this training cycle """

    def __init__(self, modelPath=None, inputSize=[128,128,1]): 
        # print('modepath:',modelPath)

        rs  = data_process.transform.inputResize(inputSize)
        it  = tf.ToTensor()

        self.img_tf = util.Compose([rs, it])
        self.network = torch.load(modelPath)
        self.network = self.network.to(device)
        self.network.eval()

    def __call__(self, imgPathList, inputSize):
        imgList = []
        # print('imgPathList:',imgPathList)

        #### 3. 直接传入图像img
        if not isinstance(imgPathList, np.ndarray) or imgPathList.shape[0] <= 0 or imgPathList.shape[1] <= 0:
            print("img none!\n")
        data = self.img_tf(imgPathList)

        # ## 2. 集成车牌检测+车牌点回归
        # if inputSize == [128, 128, 1]:
        #     img = cv2.imread(imgPathList, 0)
        # else:
        #     img = cv2.imread(imgPathList)
        
        # # print(img.shape)
        # if not isinstance(img, np.ndarray) or img.shape[0] <= 0 or img.shape[1] <= 0:
        #     print("img none!\n")
        #     # continue
        # #img = img.transpose((2, 0, 1))
        # data = self.img_tf(img)
        data = torch.unsqueeze(data, 0)
        # print('data:', data)   # ok
        data = data.to(device)
        with torch.no_grad():
            out = self.network(data)            

        eyePoints = out[0].cpu().detach().numpy()
        return eyePoints


        ## 1. 单独测试landmark点回归
        # with open(str(imgPathList[0]), 'r') as fi:
        #     for line in fi:
        #         line = line.strip().split(' ')
        #         if len(line) % 8 == 1:
        #             imgList.append('/media/mario/新加卷/DataSets/ALPR/zhongdong/' + line[0])
        #         elif len(line) % 8 == 2:
        #             imgList.append('/media/mario/新加卷/DataSets/ALPR/zhongdong/' + line[0] + ' ' + line[1])
        #         else:
        #             continue
        # random.shuffle(imgList)
        # # print('imgList len:', len(imgList))

        # for curPath in imgList:
        #     print('curPath:', curPath)
        #     # curPath = 'test_img/5.jpg'
        #     if inputSize == [128, 128, 1]:
        #         img = cv2.imread(curPath, 0)
        #     else:
        #         img = cv2.imread(curPath)
            
        #     # print(img.shape)
        #     if not isinstance(img, np.ndarray) or img.shape[0] <= 0 or img.shape[1] <= 0:
        #         print("img none!\n")
        #         # continue
        #     #img = img.transpose((2, 0, 1))
        #     data = self.img_tf(img)
        #     data = torch.unsqueeze(data, 0)
        #     # print('data:', data)   # ok
        #     data = data.to(device)
        #     with torch.no_grad():
        #         out = self.network(data)
                

        #     eyePoints = out[0].cpu().detach().numpy()
        #     # return eyePoints

        #     drawImg = draw_points(img,eyePoints)
        #     # print(label)
        #     # cv2.namedWindow("img", 0)
        #     cv2.imshow("img", drawImg)
        #     cv2.waitKey(0)