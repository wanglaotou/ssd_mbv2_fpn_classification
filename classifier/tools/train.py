#coding:utf-8
from __future__ import print_function
import os,sys
import numpy as np
import random
import time
import torch
import torchvision
import torch.nn.functional as F
from torch.utils import data
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
# 内部文件
# sys.path.append("..")

from data import Dataset,Augment
from models import *
from config.config import Config
from utils import Visualizer

# 采用warmup调整学习率
from solver import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model,metric_fc,save_path,name,iter_cnt):
    """
    保存模型,包括inference与metric

    Args：
        save_path：模型存储位置
        name：保存名称
        iter_cnt：迭代次数
    """
    save_name = os.path.join(save_path, '{}_{}.pth'.format(name,iter_cnt))
    torch.save({"inference": model.state_dict(),
                "metric_fc": metric_fc.state_dict()}, 
                save_name)
    return save_name

def load_model(model,metric_fc,load_path):
    """
    加载模型

    Args:
        model: inference模型
        metric_fc: 全连接层映射
        load_path：模型保存路径
    """
    assert os.path.exists(load_path)
    checkpoint = torch.load(load_path)
    # # 加载部分模型
    # model_dict = model.state_dict()
    # pretrained_dict = resnet50.state_dict()
    # pretrained_dict = {k: v for k,v in pretrained_dict.item() if k in model_dict}
    # # 更新现有的model_dict
    # model_dict.update(pretrained_dict)
    model.load_state_dict(checkpoint['inference'],strict=False)#strict=False，忽略不匹配的键
    metric_fc.load_state_dict(checkpoint['metric_fc'],strict=False)
    return model,metric_fc

def train(model,metric_fc,device,train_loader,criterion,optimizer,epoch,opt):
    """
    
    Args:
        model: inference模型
        metric_fc: 全连接层映射
        device: ...
        criterion: loss
        optimizer: ...
        epoch: ...
        opt: 配置文件
    """
    start = time.time()
    model.train()
    metric_fc.train()
    for ii, data in enumerate(train_loader):
        data_input, label = data['image'], data['label']
        data_input = data_input.to(device)
        label = label.to(device).long()
        feature = model(data_input)
        try:
            metric_out,output = metric_fc(feature, label)
        except TypeError:
            output = metric_fc(feature)
            metric_out = output.clone()
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iters = epoch * len(train_loader) + ii

        if iters % opt.print_freq == 0:
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            metric_out = metric_out.data.cpu().numpy()
            metric_out = np.argmax(metric_out, axis=1)
            label = label.data.cpu().numpy()
            # print(output)
            # print(label)
            acc = np.mean((output == label).astype(int))
            acc_metric = np.mean((metric_out == label).astype(int))
            speed = opt.print_freq / (time.time() - start)
            start = time.time()
            time_str = time.asctime(time.localtime(time.time()))
            print('[{}] train epoch: {}, iter: {}, lr:{}, {:.1f} iters/s, loss: {}, acc: {:.4f},metric acc: {:.4f}'
                    .format(time_str,epoch,ii,optimizer.param_groups[0]['lr'],speed,loss.item(),acc,acc_metric))
            if opt.display:
                visualizer.display_current_results(iters, loss.item(), name='train_loss')
                visualizer.display_current_results(iters, acc, name='train_acc')

def valid(model,metric_fc,device,test_loader,criterion,epoch,opt):
    """
    
    Args:
        model: inference模型
        metric_fc: 全连接层映射
        device: ...
        criterion: loss
        epoch: ...
        opt: 配置文件
    """
    start = time.time()
    model.eval()
    metric_fc.eval()
    test_loss = 0
    correct = 0
    correct_metric = 0
    with torch.no_grad(): 
        for data in test_loader:
            data_input, label = data['image'], data['label']
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            try:
                metric_out,output = metric_fc(feature, label)
            except TypeError:
                output = metric_fc(feature)
                metric_out = output.clone()
            test_loss += criterion(output, label)
            
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            metric_out = metric_out.data.cpu().numpy()
            metric_out = np.argmax(metric_out, axis=1)
            label = label.data.cpu().numpy()
            correct += np.sum(output == label)
            correct_metric += np.sum(metric_out == label)

    speed = opt.print_freq / (time.time() - start)
    test_loss /= len(test_loader)
    time_str = time.asctime(time.localtime(time.time()))
    print('[{}] valid epoch: {}, {:.1f} iters/s, loss {}, acc: {:.4f}, metric acc: {:.4f}'
            .format(time_str,epoch,speed,test_loss,correct/len(test_loader.dataset),
                    correct_metric/len(test_loader.dataset)))
    if opt.display:
        visualizer.display_current_results(epoch, correct/len(test_loader.dataset), name='valid_acc')
        visualizer.display_current_results(epoch, correct_metric/len(test_loader.dataset), name='valid_metric_acc')
    return correct/len(test_loader.dataset)

if __name__ == "__main__":
    device = DEVICE
    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    
    train_dataset = Dataset(csv_file=opt.train_file,root_dir=opt.train_root,
                        transform=Augment(np.array(opt.train_key),opt.ipt_shape))
    train_loader = data.DataLoader(train_dataset, 
                                batch_size=opt.train_batchsize,
                                shuffle=True, num_workers=32,
                                pin_memory=True)
    test_dataset = Dataset(csv_file=opt.test_file,root_dir=opt.test_root,
                        transform=Augment(np.array(opt.test_key),opt.ipt_shape))
    test_loader = data.DataLoader(test_dataset, 
                                batch_size=opt.test_batchsize,
                                shuffle=True, num_workers=16,
                                pin_memory=True)


    print('{} train iters per epoch:'.format(len(train_loader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    if opt.backbone == 'mobilenetV1':
        model = mobilenet(version='v1')
    if opt.backbone == 'peleenet64':
        model = peleenet()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(opt.feature_dim, opt.num_classes, s=10, m=0.2)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(opt.feature_dim, opt.num_classes, s=30, m=0.35, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(opt.feature_dim, opt.num_classes, m=4)
    else:
        metric_fc = torch.nn.Linear(opt.feature_dim, opt.num_classes)
    
    if opt.finetune:
        load_model(model,metric_fc,opt.load_path)

    print(model)
    model.to(device)
    metric_fc.to(device)
    # 多gpu使用
    #model = DataParallel(model)
    #metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    
    if opt.lr_strategy == 'step':
        scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)
    elif opt.lr_strategy == 'warmup':
        # warmpu+cosine 学习率
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,opt.max_epoch,eta_min=1e-5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1e2, total_epoch=opt.warm_up_epoch, after_scheduler=scheduler_cosine)

    start = time.time()
    valid_acc_max = 0.8
    for i in range(opt.max_epoch):
        train(model,metric_fc,device,train_loader,criterion,optimizer,i,opt)
        valid_acc = valid(model,metric_fc,device,test_loader,criterion,i,opt)
        
        visualizer.display_current_results(i, optimizer.param_groups[0]['lr'], name='learning_rate')

        scheduler.step()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model,metric_fc,opt.checkpoints_path,opt.save_model_name,i)

        if valid_acc>valid_acc_max:
            save_model(model,metric_fc,opt.checkpoints_path,opt.save_model_name,i)
            valid_acc_max = valid_acc
            print("best valid acc {} at {}".format(valid_acc_max,i))
        
