'''
@Author: Jiangtao
@Date: 2019-09-20 16:40:34
@LastEditors: Jiangtao
@LastEditTime: 2019-09-21 18:17:59
@Description: 
'''
import glob
import cv2
import os, sys
import numpy as np 
import pickle
import copy
import time
import random
import xml.dom.minidom
from dataAug_pts import *

root_dir = '/media/mario/新加卷/DataSets/ALPR/zhongdong'

save_dir = "landmark/lmark_train"
pos_save_dir = os.path.join(root_dir,save_dir)

if not os.path.exists(pos_save_dir):
    os.makedirs(pos_save_dir)

f1 = open(os.path.join(root_dir, 'landmark/train.txt'), 'w')

with open(os.path.join(root_dir, 'landmark/lmark_train.txt') ,'r') as f:
    lines = f.readlines()

# 统计真正干净的数据
valid_data = 0
for line in lines:

    annotations = line.strip().split(' ')
    if len(annotations) % 12 == 2:
        jpgfile = annotations[0]
        # print(jpgfile)
        filename = os.path.basename(jpgfile)
        filename = os.path.splitext(filename)[0]
        
        img_ori = cv2.imread(os.path.join(root_dir,jpgfile))
        height, width, channel = img_ori.shape
        n = int(annotations[1])
        
        points = []

        if n == 1:
            for i in range(12):
                if i < 4:
                    points.append(int(annotations[2+i]))
                if i >= 4:
                    points.append(float(annotations[2+i]))

            points = np.array(points)
            pts_ori = points.reshape((6,2))
            
            boxes = pts_ori[0:2]
            pts_ori = pts_ori[2:]
        
        # elif n == 2:
        #     for i in range(24):
        #         if i < 4 or (i > 11 and i < 16):
        #             points.append(int(annotations[2+i]))
        #         if (i >= 4 and i < 12) or i >= 16:
        #             points.append(float(annotations[2+i]))

        #     points = np.array(points)
        #     pts_ori = points.reshape((12,2))
            
        #     boxes_1 = pts_ori[0:2]
        #     boxes_2 = pts_ori[6:8]
        #     pts_ori_1 = pts_ori[2:6]
        #     pts_ori_2 = pts_ori[8:]

        #     flag = random.choice([True,False])

        #     if flag:
        #         boxes = boxes_1
        #         pts_ori = pts_ori_1
        #     else:
        #         boxes = boxes_2
        #         pts_ori = pts_ori_2
        # elif n == 3:
        #     for i in range(36):
        #         if i < 4 or (i > 11 and i < 16) or (i > 23 and i < 28):
        #             points.append(int(annotations[2+i]))
        #         if (i >= 4 and i < 12) or (i >= 16 and i < 24) or (i >= 28):
        #             points.append(float(annotations[2+i]))

        #     points = np.array(points)
        #     pts_ori = points.reshape((18,2))
            
        #     boxes_1 = pts_ori[0:2]
        #     boxes_2 = pts_ori[6:8]
        #     boxes_3 = pts_ori[12:14]
        #     pts_ori_1 = pts_ori[2:6]
        #     pts_ori_2 = pts_ori[8:12]
        #     pts_ori_3 = pts_ori[14:]

        #     # flag = random.choice([True,False])
        #     flag = random.random()

        #     if flag < 0.33:
        #         boxes = boxes_1
        #         pts_ori = pts_ori_1
        #     elif (flag >= 0.33 and flag < 0.67):
        #         boxes = boxes_2
        #         pts_ori = pts_ori_2
        #     else:
        #         boxes = boxes_3
        #         pts_ori = pts_ori_3
          
        else:
            continue
    if len(annotations) % 12 == 3:
        jpgfile = annotations[0] + ' ' + annotations[1]
        # print(jpgfile)
        filename = os.path.basename(jpgfile)
        filename = os.path.splitext(filename)[0]
        
        img_ori = cv2.imread(os.path.join(root_dir,jpgfile))
        height, width, channel = img_ori.shape
        n = int(annotations[2])
        
        points = []

        if n == 1:
            for i in range(12):
                if i < 4:
                    points.append(int(annotations[3+i]))
                if i >= 4:
                    points.append(float(annotations[3+i]))

            points = np.array(points)
            pts_ori = points.reshape((6,2))
            
            boxes = pts_ori[0:2]
            pts_ori = pts_ori[2:]
        
        # elif n == 2:
        #     for i in range(24):
        #         if i < 4 or (i > 11 and i < 16):
        #             points.append(int(annotations[3+i]))
        #         if (i >= 4 and i < 12) or i >= 16:
        #             points.append(float(annotations[3+i]))

        #     points = np.array(points)
        #     pts_ori = points.reshape((12,2))
            
        #     boxes_1 = pts_ori[0:2]
        #     boxes_2 = pts_ori[6:8]
        #     pts_ori_1 = pts_ori[2:6]
        #     pts_ori_2 = pts_ori[8:]

        #     flag = random.choice([True,False])

        #     if flag:
        #         boxes = boxes_1
        #         pts_ori = pts_ori_1
        #     else:
        #         boxes = boxes_2
        #         pts_ori = pts_ori_2
        # elif n == 3:
        #     for i in range(36):
        #         if i < 4 or (i > 11 and i < 16) or (i > 23 and i < 28):
        #             points.append(int(annotations[3+i]))
        #         if (i >= 4 and i < 12) or (i >= 16 and i < 24) or (i >= 28):
        #             points.append(float(annotations[3+i]))

        #     points = np.array(points)
        #     pts_ori = points.reshape((18,2))
            
        #     boxes_1 = pts_ori[0:2]
        #     boxes_2 = pts_ori[6:8]
        #     boxes_3 = pts_ori[12:14]
        #     pts_ori_1 = pts_ori[2:6]
        #     pts_ori_2 = pts_ori[8:12]
        #     pts_ori_3 = pts_ori[14:]

        #     # flag = random.choice([True,False])
        #     flag = random.random()

        #     if flag < 0.33:
        #         boxes = boxes_1
        #         pts_ori = pts_ori_1
        #     elif (flag >= 0.33 and flag < 0.67):
        #         boxes = boxes_2
        #         pts_ori = pts_ori_2
        #     else:
        #         boxes = boxes_3
        #         pts_ori = pts_ori_3
        
        else:
            continue

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # for i in range (4):
    #     cv2.circle(img_ori,(int(pts_ori[i][0]),int(pts_ori[i][1])),3,(0,255,0),-1)
    #     cv2.putText(img_ori,str(i),(int(pts_ori[i][0]),int(pts_ori[i][1])), font, 0.4, (255, 255, 255), 1)
    # cv2.imshow('img',img_ori)
    # cv2.waitKey(0)
    
    # 生成正样本
    count = 0
    pos_num = 1
    total_each_num = 6
    while pos_num < total_each_num:
        count += 1
        if count > 100000:
            break
        
        img = copy.deepcopy(img_ori)
        pts = copy.deepcopy(pts_ori)

        img,pts = randomAug(img,pts)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # for i in range (4):
        #     cv2.circle(img,(int(pts[i][0]),int(pts[i][1])),3,(0,255,0),-1)
        #     cv2.putText(img,str(i),(int(pts[i][0]),int(pts[i][1])), font, 0.4, (255, 255, 255), 1)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

        h,w = img.shape[0:2]
        
        # print(pts[:,0])
        pts[:,0] = pts[:,0] / float(w)
        pts[:,1] = pts[:,1] / float(h)


        resized_im = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        save_file = os.path.join(pos_save_dir, "{}_pos_{}.jpg".format(filename,pos_num))
        cv2.imwrite(save_file, resized_im)

        f1.write(save_file + ' ')

        for i in range(pts.shape[0]):
            x = pts[i][0]
            y = pts[i][1]
            f1.write('%.8f %.8f ' %(x, y))
        f1.write('\n')

        pos_num +=1
        
    valid_data += 1
    # time.sleep(10)
    print("{} images done, pos: {}".format(valid_data, pos_num))

f1.close()



