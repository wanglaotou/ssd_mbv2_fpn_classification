#coding:utf-8
import os
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageEnhance
import numpy as np
import torch as t
import torchvision.transforms as transforms
import random
import torch, torchvision

# 默认输入网络的图片大小
IMAGE_H = 128
IMAGE_W = 128

transforms=transforms.Compose([
    # transforms.Resize(IMAGE_W),  #缩放图片（Image）,保持长宽比不变，最短边为224像素
    transforms.ToTensor(), #将图片Image转换成Tensor，归一化至【0,1】
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])  #标准化至【-1,1】，规定均值和方差
])

class Dataset_belt(Dataset):      # 新建一个数据集类，并且需要继承PyTorch中的data.Dataset父类
    def __init__(self, imagelistPath,dataenhance=True,data_transform=None):          # 默认构造函数，传入数据集类别（训练或测试），以及数据集路径
        self.list_img = []
        self.list_label = []
        self.data_size = 0
        self.transform = data_transform
        self.dataenhance = dataenhance
        f = open(imagelistPath,'r').readlines()
        self.count = len(f)
        for info in f:
            
            self.imagepath = info.split(' ')[0]
            self.list_img.append(self.imagepath)
            self.list_label.append(int(info.split(' ')[1].strip('\n')))
        
    def __len__(self):
        return self.count               # 返回数据集大小

    def __getitem__(self, item):        # 重载data.Dataset父类方法，获取数据集中数据内容
        pil_img=Image.open(self.list_img[item]).convert('RGB')
        pil_img = pil_img.resize((IMAGE_W, IMAGE_H))
        dataenhance_jubu = random.randint(0,10)
        dataenhance_night = random.randint(0,10)
        if dataenhance_jubu > 7 and self.dataenhance:
            pil_img = jubu(pil_img)
        if dataenhance_night >5 and self.dataenhance: 
            pil_img = night(pil_img)

            # if dataenhance_pro > 8:
            #     pil_img = jubu(pil_img)
            # if dataenhance_pro < 8:
            #     select = random.randint(1,5)
            #     if select == 1:
            #         pil_img = ImageEnhance.Color(pil_img).enhance(random.randint(1,8)/4.0) #0.5-2
            #     if select == 2:
            #         pil_img = ImageEnhance.Brightness(pil_img).enhance(random.randint(1,8)/4.0)#0.5-2
            #     if select == 3:
            #         pil_img = ImageEnhance.Contrast(pil_img).enhance(random.randint(1,8)/4.02) #0.5-2
            #     if select == 4:
            #         pil_img = ImageEnhance.Sharpness(pil_img).enhance(random.randint(1,12)/4.0) #0.5-3
            #     if select == 5:
            #         pil_img = yinyang(pil_img)

        if self.transform:
            data=self.transform(pil_img)
        label = self.list_label[item]
        return data,label

def yinyang(image):
    bili = random.randint(1,99)/100.0
    d = random.randint(70,90)/100.0
    image_0 = Image.new('RGB',(int(np.shape(image)[1]*bili),np.shape(image)[0]),'white')
    target = Image.new('RGB', (np.shape(image)[1], np.shape(image)[0]),'black')
    target.paste(image_0, (0, 0, int(np.shape(image)[1]*bili), np.shape(image)[0]))
    img2 = Image.blend(target, image, d)#0.7-0.9
    return img2
def jubu(img):
    ranint = random.randint(2,5)
    for i in range(ranint):
        x1 = random.randint(0,np.shape(img)[1])
        y1 = random.randint(0,np.shape(img)[0])
        width = random.randint(0,80)
        height = random.randint(0,80)
        p = Image.new('RGBA', (width,height), (0,0,0))
        img.paste(p,(x1,y1))
    return img
def night(img):
    r, g, b = img.split()
    p = Image.merge("RGB",[r,r,r])
    return p

# im1 = Image.open('/home/workspace/licong/data/BeltData/belt_off/beijingbelt_53591.jpg')
# img = night(im1)
# img.save('test.jpg')
# if __name__ == "__main__":

#     zhatu_dataset = Dataset_belt('../data/imagelist.txt',dataenhance=True,data_transform = transforms)
#     print(len(zhatu_dataset))
#     dataloader = DataLoader(zhatu_dataset, batch_size=1, shuffle=True, num_workers=10)
#     for img, label in dataloader:
#         img = torchvision.utils.make_grid(img).numpy()
#         img = np.transpose(img,(1,2,0))
#         img += np.array([1,1,1])
#         img *= 127.5
#         img = img.astype(np.uint8)
#         im = Image.fromarray(img)
#         im.save("test.jpg")    
#         print(label)
#         exit()
#     print(len(zhatu_dataset))
#     for image,label in zhatu_dataset:
#         print(image.size(),label)