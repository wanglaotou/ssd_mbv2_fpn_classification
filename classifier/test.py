#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      :train.py
@Date      :2020/03/12 10:05:08
@Author    :mrwang
@Version   :1.0
'''

from torch.utils.data import DataLoader
from Data_Loading import ListDataset
from network import Net
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import os 
import cv2 as cv
from visdom import Visdom
from PIL import Image,ImageEnhance
# from visualizer import Visualizer
IMAGE_W = 128

transforms=transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize(IMAGE_W),  #缩放图片（Image）,保持长宽比不变，最短边为224像素
    transforms.ToTensor(), #将图片Image转换成Tensor，归一化至【0,1】
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])  #标准化至【-1,1】，规定均值和方差

])
 
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():    
        for img, label in test_loader: 
            img, label = torch.autograd.Variable(img).cuda(), torch.autograd.Variable(label).cuda()  
            output = model(img)
            op = model.predict(img)
            print('op:', op)
            print('label, output:', label, output)
            test_loss += criterion(output, label)
            pred = output.max(1, keepdim=True)[1]
            print('pred:', pred)
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    print("\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.06f}%)".format(
        test_loss, correct, len(test_loader.dataset),
        100.*correct/len(test_loader.dataset)))



def testImage(modelPath,testimage):
    wrong_pre = 0.0
    model = torch.load(modelPath).to(device)
    model.eval()  
    testlist = open(testimage).readlines()
    total = float(len(testlist))
    for testimage in testlist: 
        # pil_img=Image.open(testimage.split(' ')[0]).convert('RGB')
        # pil_img = pil_img.resize((128, 128))
        pil_img = cv.imread(testimage.split(' ')[0])
        pil_img = cv.resize(pil_img,(128,128))

        imgblob = transforms(pil_img)
        imgblob = imgblob.to(device)
        imgblob = imgblob.unsqueeze(0)
        output = model(imgblob)
        out = output.view(-1, 2)
        pred = out.argmax().cpu().numpy()
        
        if not int(testimage.split(' ')[1].strip('\n')) == pred:
            wrong_pre +=1
            print(testimage.split(' ')[1].strip('\n'),pred)
    correct = (total-wrong_pre)/total
    print(correct)

def testVideo(modelPath,videoPath):
    model = torch.load(modelPath).to(device)
    model.eval()
    cap=cv.VideoCapture(videoPath)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps =cap.get(cv.CAP_PROP_FPS)
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    outVideo = cv.VideoWriter('output.avi',fourcc, fps, size)    
    while(cap.isOpened()): 
        ret, frame = cap.read() #263;204;630;578
        if ret ==False:
            break
        pil_img = frame.copy()
        pil_img = pil_img[int(204*(1-0.15)):int(578+204*0.25),630+30:630+30+(int(578+204*0.25)-int(204*(1-0.15)))]
        cv.rectangle(frame,(630+30,int(204*(1-0.15))),(630+30+(int(578+204*0.25)-int(204*(1-0.15))),int(578+204*0.25)),(0, 255, 0), 2)
        pil_img = cv.resize(pil_img,(128,128))
        imgblob = transforms(pil_img)
        imgblob = imgblob.to(device)
        imgblob = imgblob.unsqueeze(0)
        output = model(imgblob)
        out = output.view(-1, 2)
        score = out.cpu().detach().numpy()
        pred = out.argmax().cpu().numpy()
        if pred == 1 :
            pre_string = 'model-1:plate_on score:'+str(score[0][1])
        if pred == 0 :
            pre_string = 'model-1:plate_off score:'+str(score[0][1])
        cv.putText(frame, pre_string, (0,40), cv.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2, cv.LINE_AA)
        outVideo.write(frame)
        # cv.imwrite('test.jpg',frame)
    cap.release()
    outVideo.release()

if __name__ == '__main__':
    display = 100
    modelPath = './colormodel/color_epoch29.pth' 
    workers = 16 
    batch_size = 8         


    root_dir = "/media/mario/新加卷/DataSets/ALPR/zhongdong/classification/classifier"
    anno_dir = os.path.join(root_dir, 'data')
    test_path = os.path.join(anno_dir, 'color_test.txt')

    device = torch.device('cuda')  

    testloader = torch.utils.data.DataLoader(ListDataset(test_path), batch_size=batch_size, shuffle=False)

    model = torch.load(modelPath)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    test(model = model,
        test_loader = testloader,
        criterion = criterion)

