#coding:utf-8
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torchstat import stat

class Net(nn.Module):
    def __init__(self, is_train=False):
        super(Net, self).__init__()
        self.is_train = is_train

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Linear(512, 2)   ## lan classification
        # self.fc = nn.Linear(512, 4)     ## color classification

    def forward(self, x):
        x = self.model(x)
        # print('x1 size:', x.size())
        x = x.view(-1, 512)
        # print('x2 size:', x.size())
        x = self.fc(x)
        # print('x3 size:', x.size())
        return x
    
    def predict(self,x):
        pred = F.softmax(self.forward(x))
        return pred

if __name__ == '__main__':  
    model = Net(is_train=False)  
    stat(model,(3, 128, 128))