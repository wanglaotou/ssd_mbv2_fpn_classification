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
from clsnetwork import Net
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

def train(model, dataloader, criterion, optimizer, display):
    model.train()                               
    for idx,(img, label) in enumerate(dataloader):
        img, label = torch.autograd.Variable(img).cuda(), torch.autograd.Variable(label).cuda()  
        optimizer.zero_grad()  
        out = model(img)   
        # print('out, label:', out.size(), label.size())  # torch.Size([32, 2]) torch.Size([64])           
        loss = criterion(out, label)    
        loss.backward()                 
        optimizer.step()                
        if idx % display == 0:
            print('train_loss {0}'.format(loss))
            # visualizer.display_current_results(idx, loss.item(), name='train_loss')
            # viz.line(X=np.array([idx]),Y=np.array([loss.cpu()]),win=win,update='append')     
def test(model, test_loader, criterion,lr):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():    
        for img, label in test_loader: 
            img, label = torch.autograd.Variable(img).cuda(), torch.autograd.Variable(label).cuda()  
            output = model(img)
            test_loss += criterion(output, label)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(test_loader)
    print("\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.06f}%),Learing rate:{}".format(
        test_loss, correct, len(test_loader.dataset),
        100.*correct/len(test_loader.dataset),lr))



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
    modelPath = './lanmodel/plate_epoch29.pth' 
    SAVE_PATH = './lanmodel/'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    # train_datalist = './data/torch_train.txt'
    # test_datalist = './data/torch_test.txt'
    workers = 16           
    batch_size = 128        
    BASE_LR = 0.001              
    epoch = 30
    is_pretrain = False

    root_dir = "/media/mario/新加卷/DataSets/ALPR/zhongdong"
    save_dir = os.path.join(root_dir,'classification')
    anno_dir = os.path.join(save_dir, 'anno_store')
    train_path = os.path.join(anno_dir, 'lantrain.txt')
    val_path = os.path.join(anno_dir, 'lanval.txt')

    
    # x,y=0,0
    # visualizer = Visualizer()

    device = torch.device('cuda')

    # train_datafile = Dataset_belt(train_datalist,dataenhance=True,data_transform = transforms)# 实例化一个数据集
    # train_dataloader = DataLoader(train_datafile, batch_size=batch_size, shuffle=True, num_workers=workers)     # 用PyTorch的DataLoader类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果

    # test_datafile = Dataset_belt(test_datalist,dataenhance=False,data_transform = transforms)# 实例化一个数据集
    # test_dataloader = DataLoader(test_datafile, batch_size=batch_size, shuffle=True, num_workers=workers)     # 用PyTorch的DataLoader类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果
    
    
    # batch_size = 32
    trainloader = torch.utils.data.DataLoader(ListDataset(train_path), batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(ListDataset(val_path), batch_size=batch_size, shuffle=True)

    # print('Dataset loaded! length of train set is {0}'.format(len(train_datafile))) 
    if is_pretrain:
        model = torch.load(modelPath)
    else:
        model = Net()
    ## EfficientNet
    # model = EfficientNet.from_pretrained('efficientnet-b7')

    # feather = model._fc.in_features
    # model._fc = nn.Linear(in_features=feather,out_features=2,bias=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=5e-4) 
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)
    for i in range(epoch):
        lr = optimizer.param_groups[0]['lr']

        ## training procedure
        train(model = model,
        optimizer = optimizer,
        criterion = criterion,
        dataloader = trainloader,
        display = display
        )
        lr_scheduler.step()

        ## testing procedure
        test(model = model,
        test_loader = testloader,
        criterion = criterion,
        lr = lr)
        
        save_model_name = os.path.join(SAVE_PATH,"lan_epoch{}.pth".format(i))
        torch.save(model, save_model_name)