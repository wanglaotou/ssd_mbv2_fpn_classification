'''
@Author: Jiangtao
@Date: 2019-09-21 10:04:02
@LastEditors: Jiangtao
@LastEditTime: 2019-09-21 10:16:25
@Description: 
'''
import glob
import cv2
import os
import numpy as np 
import pickle
import copy
import time
import random
import xml.dom.minidom
from dataAug_pts import *

txt_dir = '/media/mario/新加卷/DataSets/ALPR/zhongdong/landmark/test.txt'

with open(txt_dir, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)

for line in lines:

    annotations = line.strip().split(' ')
    if len(annotations) % 8 == 1:
        jpgfile = annotations[0]
        print('jpgfile:', jpgfile)
        img_ori = cv2.imread(jpgfile)
        points = []
        for i in range(8):
            points.append(float(annotations[1+i]))
    elif len(annotations) % 8 == 2:
        jpgfile = annotations[0] + ' ' + annotations[1]
        print('jpgfile:', jpgfile)
        img_ori = cv2.imread(jpgfile)
        points = []
        for i in range(8):
            points.append(float(annotations[2+i]))

    points = np.array(points)
    pts_ori = points.reshape((4,2))*128

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range (4):
        cv2.circle(img_ori,(int(pts_ori[i][0]),int(pts_ori[i][1])),3,(0,255,0),-1)
        cv2.putText(img_ori,str(i),(int(pts_ori[i][0]),int(pts_ori[i][1])), font, 0.4, (255, 255, 255), 1)
    cv2.imshow('img',img_ori)
    cv2.waitKey(0)
