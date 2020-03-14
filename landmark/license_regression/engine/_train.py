'''
@Author: Jiangtao
@Date: 2019-08-07 10:42:06
@LastEditors: Jiangtao
@LastEditTime: 2019-09-23 15:08:21
@Description: 
'''

import torch
import cv2
import sys
import numpy as np
from torchvision import transforms as tf
import data_process
import util
from network import backbone
import time
from collections import OrderedDict

import loss as ls


# import file_operation as fo
#import pts_operation as po

__all__ = ['TrainingEngine']

# class classificDataset(data_process.DatasetWithAngleMulti3):
#     def __init__(self, imgListPath, inputSize):
#         rs  = data_process.transform.inputResize(inputSize)
#         it  = tf.ToTensor()
#         img_tf = util.Compose([rs, it])
#         label_tf = util.Compose([it])
#         super(classificDataset, self).__init__(imgListPath, inputSize, img_tf, label_tf, imgChannel=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainingEngine():

    """ This is a custom engine for this training cycle """

    def __init__(self, modelPath=None, imgListPath=None, classNum=8, batchSize=32, workers=4, imgChannel = 1, inputSize=[128, 128, 1]):

        rs  = data_process.transform.inputResize(inputSize)
        it  = tf.ToTensor()
        img_tf = util.Compose([rs, it])
        label_tf = util.Compose([it])

        TrainDataset = data_process.DatasetWithAngleMulti3(imgListPath, inputSize, img_tf, label_tf,\
                                                            imgChannel=imgChannel,isTrain='train')

        self.dataLoader = torch.utils.data.DataLoader(TrainDataset, batch_size=batchSize, shuffle=True,\
                                               collate_fn=TrainDataset.collate_fn, num_workers=workers,\
                                               pin_memory=True)  # note that we're passing the collate function here
     

        testDataset = data_process.DatasetWithAngleMulti3(imgListPath, inputSize, img_tf, label_tf,\
                                                            imgChannel=imgChannel,isTrain='val')

        self.testdataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, shuffle=True,\
                                               collate_fn=testDataset.collate_fn, num_workers=workers,\
                                               pin_memory=True)  # note that we're passing the collate function here       


        self.learning_rate = 0.0001
        self.momentum = 0.9
        self.decay = 0.0005

        print("modelPath: ", modelPath)

        if modelPath is None:
            
            self.network = backbone.licenseBone()
            
            print('have loaded old model')

        else:
            self.network = torch.load(modelPath)

        self.network = self.network.to(device)

        # 设置需要训练的参数
        for k,v in self.network.named_parameters():
            
            v.requires_grad=True


        for name, param in self.network.named_parameters():
            print(name,param.requires_grad)
        
        self.loss = None
        self.lossFn = ls.l2lossWithMulti3()
        self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

    def train_batch_with_ang_multi3(self, data, idx):
        # self.network.faceBone.train()
        imgBatch = data[0].to(device)
        
        label1 = data[1].to(device)


        #print(imgBatch.size(), label1.size(), label2.size())  
        out = self.network(imgBatch)
        # print(imgBatch.size(), label1.size(), label2.size(), out[0].size(), out[1].size())   
        self.loss = self.lossFn(out[0], label1)
        # self.loss = self.lossFn(out[0], label1, out[1], label2, out[2], label3, out[3], label4)

        self.loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        if idx % 50 == 0:
            print("train: idx:{},loss:{}".format(idx,self.loss.data))

    def val_batch_with_ang_multi3(self):

        lossList = []
        correctList = []

        for idx_, data_ in enumerate(self.testdataLoader):

            imgBatch = data_[0].to(device)

            label1 = data_[1].to(device)
            # label2 = data_[2].to(device)
            # label3 = data_[3].to(device)
            # label4 = data_[4].to(device)


            with torch.no_grad():

                out = self.network(imgBatch)

                # _, preds = out[4].max(1)
                # correct = preds.eq(label3.long()).sum()
                # correct = correct.float() / imgBatch.shape[0]

                loss = self.lossFn(out[0], label1)
                # loss = self.lossFn(out[0], label1, out[1], label2, out[2], label3, 
                #                         out[3], label4)

                # correctList.append(correct.item())
                lossList.append(loss.item())


        # correctArray = np.array(correctList)
        lossArray = np.array(lossList)

        # print("validation: idx_:{},loss:{},correct:{}".format(idx_,lossArray.mean(),correctArray.mean()))
        print("validation: idx_:{},loss:{}".format(idx_,lossArray.mean()))


    def __call__(self):
     
        epoch = 50
        for i in range(epoch):

            for idx, data in enumerate(self.dataLoader):
                # print(data.shape)
                self.train_batch_with_ang_multi3(data, idx)

                if (idx % int(0.5 * len(self.dataLoader))) == 0 :   #训练的同时每隔半个epoch迭代验证一次。

                    self.val_batch_with_ang_multi3()

            print("epoch:{} ,loss:{} ".format(i,self.loss.data))

            savePath = "./models/" + "multi_epoch_" + str(i) + "_landmark.pkl"
            torch.save(self.network, savePath)
            epoch += 1


