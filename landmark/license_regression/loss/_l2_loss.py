'''
@Author: Jiangtao
@Date: 2019-08-07 10:42:06
@LastEditors: Jiangtao
@LastEditTime: 2019-09-23 15:02:39
@Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

__all__ = ['l2Loss', 'smoothL1Loss', 'l2lossWithMulti3']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 支持多分类和二分类
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
 
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
 
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
 
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
 
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
 
        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
 
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
 
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
 
        gamma = self.gamma
 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
        
class smoothL1Loss(nn.Module):
    def __init__(self):
        super(smoothL1Loss, self).__init__()

        self.outLoss = nn.SmoothL1Loss(reduce=False)
        self.outLoss = self.outLoss.to(device)

    def forward(self, output, target):
       # print(output.size(), target.size())
        if isinstance(output, (list, tuple)):
            loss = 0
            for i in range(len(output)):
                loss += self.outLoss(output[i], target).sum()
                print("loss", loss)
            return loss / len(output)
        else:
            loss = self.outLoss(output, target)
            lossSum = loss.sum() / output.size()[0]
            #print(output.size(), target.size(), loss.size(), lossSum, output.size()[0])
            return lossSum

class l2lossWithMulti3(nn.Module):
    def __init__(self):
        super(l2lossWithMulti3, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss(reduce=False)
        self.mse = self.mse.to(device)
        self.FocalLoss = FocalLoss(3)
        self.UseFl = False

    def forward(self, output1, target1):

        # print(output1.shape)
        # print(target1.shape)
        loss = 0
        # 眼睛
        loss += (((self.mse(output1, target1))).sum())

        # 损失均值化
        loss = loss / float(output1.size()[0])
        return loss