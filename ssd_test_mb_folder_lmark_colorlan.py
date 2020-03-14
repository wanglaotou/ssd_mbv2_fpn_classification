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
net.load_state_dict(torch.load('weights_zhongdong/ssd_mobilenetv2_fpn_20200305_addmiss/mobilenetv2_290000.pth'))  # test mobilenet backbone
net.eval()
from clsnetwork import Net

# image = cv2.imread('./data/example.jpg', cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
# %matplotlib inline
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
# here we specify year (07 or 12) and dataset ('test', 'val', 'train') 
root_test = '/media/mario/新加卷/DataSets/ALPR/zhongdong'
save_root = '/media/mario/新加卷/DataSets/videosrc_Results/zhongdong'

ftr = open('eval/plate_mb_result_zhongdong_fpn_29w_thre045_lmark_colorlan_0314.txt','w')
test_imgpath = os.path.join(save_root,'plate_mb_result_zhongdong_fpn_29w_thre045_lmark_colorlan_0314')
color_dict = {'0': 'yellow_card', '1': 'white_card', '2': 'red_card', '3': 'green_card'}
lan_dict = {'0': 'Double_column', '1': 'Single_column'}
# yellow_card:0
# white_card:1
# red_card:2
# green_card:3
# Double_column:0
# Single_column:1
if not os.path.exists(test_imgpath):
    os.mkdir(test_imgpath)

color_model_path = 'classifier/colormodel/color_epoch29.pth'
lan_model_path = 'classifier/lanmodel/lan_epoch29.pth'
color_model = (torch.load(color_model_path))
lan_model = (torch.load(lan_model_path))
color_model = color_model.to(device)
lan_model = lan_model.to(device)
image_dir = '/media/mario/新加卷/DataSets/videosrc/zhongdong/test'
img_paths = [el for el in paths.list_images(image_dir)]
random.shuffle(img_paths)
for imgpath in img_paths:
    print(imgpath)
    image = cv2.imread(imgpath)
    imagename = os.path.basename(imgpath)
    save_imgpath = os.path.join(test_imgpath, imagename)
    height, width, channel = image.shape
    img_ori = copy.deepcopy(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print('rgb_image shape:', rgb_image.shape)

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
    top_k=10

    detections = y.data
    # scale each detection back up to the image64
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

    landmark = []
    vidcolor = []
    vidlan = []
    for i in range(detections.size(1)):
        j = 0
        # print('score:', detections[0,i,j,0])
        while detections[0,i,j,0] >= 0.45:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            if(score>=0.45):
                # print('display_txt:', display_txt)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                # print('coords:', coords)
                # ftr.write(imagename+' '+str(np.round(score.cpu().numpy(),3)))
                # ftr.write(' '+str(int(pt[0]))+' '+str(int(pt[1]))+' '+str(int(pt[2]))+' '+str(int(pt[3]))+'\n')
                
                w = int(pt[2]) - int(pt[0])
                h = int(pt[3]) - int(pt[1])

                # random delta x and y
                # delta_x1 = np.random.randint(int(w * 0.2), int(w * 0.5))
                # delta_y1 = np.random.randint(int(h * 0.2), int(h * 0.5))
                # delta_x2 = np.random.randint(int(w * 0.2), int(w * 0.5))
                # delta_y2 = np.random.randint(int(h * 0.2), int(h * 0.5))
                # fix delta x and y
                delta_x1 = int(w * 0.3)
                delta_y1 = int(h * 0.3)
                delta_x2 = int(w * 0.3)
                delta_y2 = int(h * 0.3)

                nx1 = max(int(pt[0]) - delta_x1,0)
                ny1 = max(int(pt[1]) - delta_y1,0)
                nx2 = min(int(pt[2]) + delta_x2, width)
                ny2 = min(int(pt[3]) + delta_y2, height)

                nw = nx2 - nx1
                nh = ny2 - ny1

                crop_img = img_ori[int(ny1): int(ny2), int(nx1): int(nx2), :]

                ## 车牌的颜色和单双栏分类预测
                color_lan_str = ''
                with torch.no_grad():
                    color_lan_img = cv2.resize(crop_img, (48, 48), interpolation=cv2.INTER_LINEAR)
                    color_lan_img = color_lan_img[:,:,::-1]
                    color_lan_img = np.asarray(color_lan_img, 'float32')
                    color_lan_img = color_lan_img.transpose((2, 0, 1))
                    color_lan_img = (color_lan_img - 127.5) * 0.0078125
                    color_lan_img = torch.FloatTensor(color_lan_img).unsqueeze(0)
                    color_lan_img = torch.autograd.Variable(color_lan_img).cuda()
                    # print('color_lan_img:', color_lan_img.shape)

                    color_model.eval()
                    lan_model.eval()
                    color_output = color_model(color_lan_img)
                    color_pred = color_output.max(1, keepdim=True)[1]
                    lan_output = lan_model(color_lan_img)

                    lan_pred = lan_output.max(1, keepdim=True)[1]
                    color_pred = str(color_pred.cpu().numpy()[0][0])
                    lan_pred = str(lan_pred.cpu().numpy()[0][0])
                    # print('color_pred:', color_pred)
                    # print('lan_pred:', lan_pred)
                    color_info = color_dict[color_pred]
                    lan_info = lan_dict[lan_pred]
                    vidcolor.append(color_info)
                    vidlan.append(lan_info)
                color_lan_str = color_info + ',' + lan_info

                ## 车牌的landmark点回归预测
                resized_im = cv2.resize(crop_img, (128, 128), interpolation=cv2.INTER_LINEAR)
                # crop_path = 'crop.jpg'
                # cv2.imwrite(crop_path, resized_im)

                ## load landmark model
                lmark_model_path = "landmark/license_regression/models/model_rgb/multi_epoch_46_landmark.pkl"
                # eng = engine.TestEngineImg(modelPath=lmark_model_path, inputSize=[128, 128, 1])

                train_rgb = True
                inputSize=[]

                if train_rgb:
                    inputSize=[128, 128, 3]
                    imgChannel = 3
                else:
                    inputSize=[128, 128, 1]
                    imgChannel = 1
                # model_path = "license_regression/models/multi_epoch_46_landmark.pkl"
                
                eng = engine.TestEngineImg(modelPath=lmark_model_path, inputSize=inputSize)
                
                # img, eyePoints = eng(crop_path)
                # eyePoints = eng(crop_path, inputSize)
                eyePoints = eng(resized_im, inputSize)
                # print(eyePoints)

                repoints = eyePoints.reshape((4,2))

                xcoord = []
                ycoord = []
                xycoords = []
                for k in range(repoints.shape[0]):
                    epoint_x = (float(repoints[k][0]) * nw + nx1)
                    epoint_y = (float(repoints[k][1]) * nh + ny1)
                    cv2.circle(image,(int(epoint_x),int(epoint_y)),5,(255,0,0),-1)
                    xcoord.append(epoint_x) 
                    ycoord.append(epoint_y)
                    coords = [epoint_x, epoint_y]
                    xycoords.append(coords)
                landmark.append(xycoords)

                xmin = min(xcoord)
                ymin = min(ycoord)
                xmax = max(xcoord)
                ymax = max(ycoord)

                cv2.putText(image,str(np.round(score.cpu().numpy(),3)),(int(xmin),int(ymin)),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
                cv2.putText(image, color_lan_str, (int(xmin),int(ymin)-20), cv2.FONT_HERSHEY_TRIPLEX, 1, color=(255,0,0))
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2) 

                # savepath = os.path.join(lmark_dir,imagename)
                # print('savepath:',savepath)
                # cv2.imwrite(savepath,image) 

            j+=1
    cv2.imwrite(save_imgpath, image)
    
    plate_num = len(landmark)
    ftr.write(imagename+' '+str(plate_num)+' ')
    for i in range(plate_num):
        lm = landmark[i]
        vc = vidcolor[i]
        vl = vidlan[i]
        ftr.write(str(vc) + ' ' + str(vl) + ' ')
        for j in range(len(lm)):
            x1, y1 = lm[j][0], lm[j][1]
            ftr.write(str(x1) + ' ' + str(y1) + ' ')
        
    ftr.write('\n')
    # image = cv2.resize(image, (1920, 1080))
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
        
