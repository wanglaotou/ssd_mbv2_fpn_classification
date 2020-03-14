'''
@Author: Jiangtao
@Date: 2019-08-07 10:42:06
@LastEditors: Jiangtao
@LastEditTime: 2019-09-23 15:10:50
@Description: 
'''
import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer


class licenseBone(nn.Module):
    def __init__(self):
        super(licenseBone, self).__init__()

        # layer0 = layer.Conv2dBatchReLU(3, 32, 3, 2)       ## 采用rgb图像进行训练
        layer0 = layer.Conv2dBatchReLU(1, 32, 3, 2)         ## 采用灰度图像进行训练
        layer1 = layer.Conv2dBatchReLU(32, 64, 3, 1)
        layer2 = layer.Conv2dBatchReLU(64, 64, 3, 2)      #32
        layer3 = layer.Conv2dBatchReLU(64, 128, 3, 1)
        layer4 = layer.Conv2dBatchReLU(128, 128, 3, 2)    #16

        self.layers = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4
        )

        layer00 = layer.Conv2dBatchReLU(128, 128, 3, 1)
        layer11 = layer.Conv2dBatchReLU(128, 256, 3, 2)
        layer22 = layer.Conv2dBatchReLU(256, 512, 3, 2)
        layer33 = layer.GlobalAvgPool2d()
        layer44 = layer.FullyConnectLayer(512, 8)

        self.layers0 = nn.Sequential(
            layer00,
            layer11,
            layer22,
            layer33,
            layer44
        )

        # self.gap = layer.GlobalAvgPool2d()
        # self.fc = layer.FullyConnectLayer(512, 8)

    def forward(self, x):

        x0 = self.layers(x)
        # print('conv5:', x0, x0.shape)  # ok
        x1 = self.layers0(x0)
        # print('x1:', x1)

        # x1 = self.gap(x1)
        # x1 = self.fc(x1)

        return [x1]
