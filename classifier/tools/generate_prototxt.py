'''
@Author: Jiangtao
@Date: 2019-08-29 15:43:40
@LastEditors: Jiangtao
@LastEditTime: 2019-08-29 16:44:51
@Description: 
'''
import sys

import h5py
caffe_root = '/home/streamax/workspace/caffe-base/'
sys.path.insert(0, caffe_root + 'python')
#sys.path.append("/home/workspace/yghan/caffe_lib/caffe/build/tools")

import caffe
from caffe import layers as L
from caffe import params as P


def test(bottom):
    conv11 = caffe.layers.Convolution(bottom, num_output=20, kernel_size=3, weight_filler={"type": "xavier"},
                                bias_filler={"type": "constant"}, param=dict(lr_mult=1))
    relu11 = caffe.layers.PRelu(conv11, in_place=True)
    pool11 = caffe.layers.Pooling(relu11, pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    
    return pool11 

def relu(bottom):
    return L.ReLU(bottom, in_place=True)

def prelu(bottom):
    return L.PReLU(bottom, in_place=True)

def drop(bottom, dropout_ratio):
    return L.Dropout(bottom, dropout_ratio=0.25, in_place=True)

def fully_connect(bottom, outputChannel):
    return L.InnerProduct(bottom, num_output=outputChannel, weight_filler=dict(type='xavier'))

def flatten(net, bottom):
    return net.blobs['bottom'].data[0].flatten()


def global_avg_pool(bottom, kernelSize=3):
    #return L.Pooling(bottom, pool=P.Pooling.AVE,stride=1, kernel_size=kernelSize)
    return L.Pooling(bottom, pool=P.Pooling.AVE, global_pooling=True)

def preluConv(bottom, outputChannel, kernelSize=(3, 5), stride=1, isTrain=True, isRelu=True):
    if len(kernelSize) == 2:
        conv = caffe.layers.Convolution(bottom, num_output=outputChannel, kernel_h=kernelSize[0], kernel_w=kernelSize[1], stride=stride, \
                                    weight_filler={"type": "xavier"},\
                                    bias_term=True, param=dict(lr_mult=1))
    elif len(kernelSize) == 1:
        conv = caffe.layers.Convolution(bottom, num_output=outputChannel, kernel_size=kernelSize[0], stride=stride, \
                                    weight_filler={"type": "xavier"},\
                                    bias_term=True, param=dict(lr_mult=1))
    else:
        pass
    if isRelu == True:
        return prelu(conv)
    else:
        return conv
def Deepwith(bottom):
    return L.ConvolutionDepthwise(bottom,num_output=32, param=dict(lr_mult=0.1))
def max_pool(bottom, kernelSize=3, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, stride_h=stride[0], stride_w=stride[1], kernel_h=kernelSize[0], kernel_w=kernelSize[1])
    # return L.Pooling(bottom, pool=P.Pooling.MAX, global_pooling=True)

def BN(bottom, isTrain=True, isRelu=False):
    use_global = False
    if isTrain == False:
        use_global=True
    bn = caffe.layers.BatchNorm(bottom, use_global_stats=use_global, in_place=True)
    scale = caffe.layers.Scale(bn, bias_term=True, in_place=True)
    if True == isRelu:
        return relu(scale)
    else:
        return scale

def basicConv(bottom, outputChannel, kernelSize=3, stride=2, isTrain=True, isRelu=True):
    
    halfKernelSize = int(kernelSize/2)
    #print("half", halfKernelSize)
    if kernelSize == 1:
        halfKernelSize=0
    if halfKernelSize == 0:
        conv = caffe.layers.Convolution(bottom, num_output=outputChannel, kernel_size=kernelSize, stride=stride, \
                                weight_filler={"type": "msra"},\
                                bias_term=False, param=dict(lr_mult=1))
    else:
        conv = caffe.layers.Convolution(bottom, num_output=outputChannel, pad=halfKernelSize, kernel_size=kernelSize, stride=stride, \
                                weight_filler={"type": "msra"},\
                                bias_term=False, param=dict(lr_mult=1)) 
    bn = BN(conv, isTrain=isTrain)
    if isRelu == True:
        return relu(bn)
    else:
        return bn


def generate_mobilenet_v1(name, intputSize=[17, 47, 1], writePath=None,isTrain=False):

    if name == 'landmark':
        net = caffe.NetSpec()
        net.data = L.Input(shape = dict(dim = [1,1,128,128]))
        net.conv0 = basicConv(net.data, 32, 3, 2, isTrain)
        net.conv1_dw = Deepwith(net.conv0)
        # backbone
             #64
        net.conv2 = basicConv(net.conv1_dw, 64, 3, 1, isTrain)

        net.gap1 = global_avg_pool(net.conv2)
        net.fc1 = fully_connect(net.gap1,8)

        
    # transform
    proto = net.to_proto()
    proto.name = name

    with open(writePath, 'w') as f:
        print("start write!\n")
        f.write(str(proto))

    net = caffe.Net(writePath, caffe.TEST)
    caffeParams = net.params
    for k in sorted(caffeParams):
        print(k)
    print(len(caffeParams))

if __name__ == '__main__':

    nettype = 'landmark' # 'pnet, onet, landmark'
    writePath = nettype + '.prototxt'
    # writePath = "onet.prototxt"
    if nettype == 'pnet':
        intputSize = [17, 47, 3]
    elif nettype == 'onet':
        intputSize = [34, 94, 3]
    elif nettype == 'landmark':
        intputSize = [128, 128, 1]
    # intputSize=[17, 47, 1]

    generate_mobilenet_v1(nettype, intputSize=intputSize, writePath=writePath)
