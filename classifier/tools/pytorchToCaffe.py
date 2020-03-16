'''
@Author: Jiangtao
@Date: 2019-08-29 15:43:40
@LastEditors: Jiangtao
@LastEditTime: 2019-08-29 17:02:15
@Description: 
'''

import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
o = 0

caffe_root = '/home/workspace/licong/caffe-base/'
sys.path.insert(0, caffe_root + 'python')
import caffe
# import layer
print(caffe.__path__)

if __name__ == '__main__':

    protofile = "./Belt_classification.prototxt"
    modelFile = "./Belt_classification.caffemodel"

    modelPath = '../model/belt1111-45000.pth'
    #caffe.set_mode_cpu()

    net = caffe.Net(protofile, caffe.TEST)
    caffeParams = net.params

    for k in caffeParams:
        print(k)
    print(len(caffeParams))
    
    network = torch.load(modelPath)
    network.eval()

    for param_tensor, value in network.state_dict().items():
        print(param_tensor)

    print((len(network.state_dict())))

    recycle = 0
    layerNum = 1
    i = 0
    sizeNum = 0
    nameDict = {
                "model.0.0.weight":"conv0,0",
                "model.0.1.weight":"conv0/scale,0",
                "model.0.1.bias":"conv0/scale,1",
                "model.0.1.running_mean":"conv0/bn,0",
                "model.0.1.running_var":"conv0/bn,1",
                "model.0.1.num_batches_tracked":"conv0/bn,2",

                "model.1.0.weight":"conv1/dw,0",
                "model.1.1.weight":"conv1/dw/scale,0",
                "model.1.1.bias":"conv1/dw/scale,1",
                "model.1.1.running_mean":"conv1/dw/bn,0",
                "model.1.1.running_var":"conv1/dw/bn,1",
                "model.1.1.num_batches_tracked":"conv1/dw/bn,2",
                "model.1.3.weight":"conv1,0",
                "model.1.4.weight":"conv1/scale,0",
                "model.1.4.bias":"conv1/scale,1",
                "model.1.4.running_mean":"conv1/bn,0",
                "model.1.4.running_var":"conv1/bn,1",
                "model.1.4.num_batches_tracked":"conv1/bn,2",    

                "model.2.0.weight":"conv2/dw,0",
                "model.2.1.weight":"conv2/dw/scale,0",
                "model.2.1.bias":"conv2/dw/scale,1",
                "model.2.1.running_mean":"conv2/dw/bn,0",
                "model.2.1.running_var":"conv2/dw/bn,1",
                "model.2.1.num_batches_tracked":"conv2/dw/bn,2",
                "model.2.3.weight":"conv2,0",
                "model.2.4.weight":"conv2/scale,0",
                "model.2.4.bias":"conv2/scale,1",
                "model.2.4.running_mean":"conv2/bn,0",
                "model.2.4.running_var":"conv2/bn,1",
                "model.2.4.num_batches_tracked":"conv2/bn,2",

                "model.3.0.weight" :"conv3/dw,0",
                "model.3.1.weight":"conv3/dw/scale,0",
                "model.3.1.bias":"conv3/dw/scale,1",
                "model.3.1.running_mean":"conv3/dw/bn,0",
                "model.3.1.running_var":"conv3/dw/bn,1",
                "model.3.1.num_batches_tracked":"conv3/dw/bn,2",
                "model.3.3.weight":"conv3,0",
                "model.3.4.weight":"conv3/scale,0",
                "model.3.4.bias":"conv3/scale,1",
                "model.3.4.running_mean":"conv3/bn,0",
                "model.3.4.running_var":"conv3/bn,1",
                "model.3.4.num_batches_tracked":"conv3/bn,2",

                "model.4.0.weight":"conv4/dw,0",
                "model.4.1.weight":"conv4/dw/scale,0",
                "model.4.1.bias":"conv4/dw/scale,1",
                "model.4.1.running_mean":"conv4/dw/bn,0",
                "model.4.1.running_var":"conv4/dw/bn,1",
                "model.4.1.num_batches_tracked":"conv4/dw/bn,2",
                "model.4.3.weight":"conv4,0",
                "model.4.4.weight":"conv4/scale,0",
                "model.4.4.bias":"conv4/scale,1",
                "model.4.4.running_mean":"conv4/bn,0",
                "model.4.4.running_var":"conv4/bn,1",
                "model.4.4.num_batches_tracked":"conv4/bn,2",

                "model.5.0.weight":"conv5/dw,0",
                "model.5.1.weight":"conv5/dw/scale,0",
                "model.5.1.bias":"conv5/dw/scale,1",
                "model.5.1.running_mean":"conv5/dw/bn,0",
                "model.5.1.running_var":"conv5/dw/bn,1",
                "model.5.1.num_batches_tracked":"conv5/dw/bn,2",
                "model.5.3.weight":"conv5,0",
                "model.5.4.weight":"conv5/scale,0",
                "model.5.4.bias":"conv5/scale,1",
                "model.5.4.running_mean":"conv5/bn,0",
                "model.5.4.running_var":"conv5/bn,1",
                "model.5.4.num_batches_tracked":"conv5/bn,2",

                "model.6.0.weight":"conv6/dw,0",
                "model.6.1.weight":"conv6/dw/scale,0",
                "model.6.1.bias":"conv6/dw/scale,1",
                "model.6.1.running_mean":"conv6/dw/bn,0",
                "model.6.1.running_var":"conv6/dw/bn,1",
                "model.6.1.num_batches_tracked":"conv6/dw/bn,2",
                "model.6.3.weight":"conv6,0",
                "model.6.4.weight":"conv6/scale,0",
                "model.6.4.bias":"conv6/scale,1",
                "model.6.4.running_mean":"conv6/bn,0",
                "model.6.4.running_var":"conv6/bn,1",
                "model.6.4.num_batches_tracked":"conv6/bn,2",

                "model.7.0.weight":"conv7/dw,0",
                "model.7.1.weight":"conv7/dw/scale,0",
                "model.7.1.bias":"conv7/dw/scale,1",
                "model.7.1.running_mean":"conv7/dw/bn,0",
                "model.7.1.running_var":"conv7/dw/bn,1",
                "model.7.1.num_batches_tracked":"conv7/dw/bn,2",
                "model.7.3.weight":"conv7,0",
                "model.7.4.weight":"conv7/scale,0",
                "model.7.4.bias":"conv7/scale,1",
                "model.7.4.running_mean":"conv7/bn,0",
                "model.7.4.running_var":"conv7/bn,1",
                "model.7.4.num_batches_tracked":"conv7/bn,2",

                "model.8.0.weight":"conv8/dw,0",
                "model.8.1.weight":"conv8/dw/scale,0",
                "model.8.1.bias":"conv8/dw/scale,1",
                "model.8.1.running_mean":"conv8/dw/bn,0",
                "model.8.1.running_var":"conv8/dw/bn,1",
                "model.8.1.num_batches_tracked":"conv8/dw/bn,2",
                "model.8.3.weight":"conv8,0",
                "model.8.4.weight":"conv8/scale,0",
                "model.8.4.bias":"conv8/scale,1",
                "model.8.4.running_mean":"conv8/bn,0",
                "model.8.4.running_var":"conv8/bn,1",
                "model.8.4.num_batches_tracked":"conv8/bn,2",

                "model.9.0.weight":"conv9/dw,0",
                "model.9.1.weight":"conv9/dw/scale,0",
                "model.9.1.bias":"conv9/dw/scale,1",
                "model.9.1.running_mean":"conv9/dw/bn,0",
                "model.9.1.running_var":"conv9/dw/bn,1",
                "model.9.1.num_batches_tracked":"conv9/dw/bn,2",
                "model.9.3.weight":"conv9,0",
                "model.9.4.weight":"conv9/scale,0",
                "model.9.4.bias":"conv9/scale,1",
                "model.9.4.running_mean":"conv9/bn,0",
                "model.9.4.running_var":"conv9/bn,1",
                "model.9.4.num_batches_tracked":"conv9/bn,2",

                "model.10.0.weight":"conv10/dw,0",
                "model.10.1.weight":"conv10/dw/scale,0",
                "model.10.1.bias":"conv10/dw/scale,1",
                "model.10.1.running_mean":"conv10/dw/bn,0",
                "model.10.1.running_var":"conv10/dw/bn,1",
                "model.10.1.num_batches_tracked":"conv10/dw/bn,2",
                "model.10.3.weight":"conv10,0",
                "model.10.4.weight":"conv10/scale,0",
                "model.10.4.bias":"conv10/scale,1",
                "model.10.4.running_mean":"conv10/bn,0",
                "model.10.4.running_var":"conv10/bn,1",
                "model.10.4.num_batches_tracked":"conv10/bn,2",

                "model.11.0.weight":"conv11/dw,0",
                "model.11.1.weight":"conv11/dw/scale,0",
                "model.11.1.bias":"conv11/dw/scale,1",
                "model.11.1.running_mean":"conv11/dw/bn,0",
                "model.11.1.running_var":"conv11/dw/bn,1",
                "model.11.1.num_batches_tracked":"conv11/dw/bn,2",
                "model.11.3.weight":"conv11,0",
                "model.11.4.weight":"conv11/scale,0",
                "model.11.4.bias":"conv11/scale,1",
                "model.11.4.running_mean":"conv11/bn,0",
                "model.11.4.running_var":"conv11/bn,1",
                "model.11.4.num_batches_tracked":"conv11/bn,2",

                "model.12.0.weight":"conv12/dw,0",
                "model.12.1.weight":"conv12/dw/scale,0",
                "model.12.1.bias":"conv12/dw/scale,1",
                "model.12.1.running_mean":"conv12/dw/bn,0",
                "model.12.1.running_var":"conv12/dw/bn,1",
                "model.12.1.num_batches_tracked":"conv12/dw/bn,2",
                "model.12.3.weight":"conv12,0",
                "model.12.4.weight":"conv12/scale,0",
                "model.12.4.bias":"conv12/scale,1",
                "model.12.4.running_mean":"conv12/bn,0",
                "model.12.4.running_var":"conv12/bn,1",
                "model.12.4.num_batches_tracked":"conv12/bn,2",

                "model.13.0.weight":"conv13/dw,0",
                "model.13.1.weight":"conv13/dw/scale,0",
                "model.13.1.bias":"conv13/dw/scale,1",
                "model.13.1.running_mean":"conv13/dw/bn,0",
                "model.13.1.running_var":"conv13/dw/bn,1",
                "model.13.1.num_batches_tracked":"conv13/dw/bn,2",
                "model.13.3.weight":"conv13,0",
                "model.13.4.weight":"conv13/scale,0",
                "model.13.4.bias":"conv13/scale,1",
                "model.13.4.running_mean":"conv13/bn,0",
                "model.13.4.running_var":"conv13/bn,1",
                "model.13.4.num_batches_tracked":"conv13/bn,2",

                "fc.weight":"fc7_new,0",
                "fc.bias":"fc7_new,1",
    }

    pytorchLayerNameList = list(nameDict.keys())
    caffeLayerNameList = list(nameDict.values())

    i = 0
    # check if all parameters in nameDict
    for param_tensor in network.state_dict():
        print(str(i)+' |', param_tensor+' |', nameDict[param_tensor])
        if param_tensor not in pytorchLayerNameList:
            print("there is some problem in nameDict")
            sys.exit()

        param = network.state_dict()[param_tensor]

        caffeLayerPara = nameDict[param_tensor]

        if "," in caffeLayerPara:
            
            caffeLayerName, caffeLayerMatNum = caffeLayerPara.strip().split(",")
            caffeLayerMatNum = int(caffeLayerMatNum)
            if caffeLayerName not in caffeParams:
                print("caffeLayerName is not in caffe")
            print(np.shape(param.cpu().data.numpy()))
            if "num_batches_tracked" in param_tensor:
                caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = np.array([1.0])
            else:
                print('view:',caffeLayerName)
                print("1:",np.shape(caffeParams[caffeLayerName][caffeLayerMatNum].data[...]))
                print("2:",np.shape(param.cpu().data.numpy()))
                
                if caffeLayerName == 'fc7_new' :
                    o+=1
                    if o ==1 :
                        caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = np.reshape(param.cpu().data.numpy(),(2,1024,1,1))
                else:
                    
                    caffeParams[caffeLayerName][caffeLayerMatNum].data[...] = param.cpu().data.numpy()
                
        i += 1
    net.save(modelFile) 
    print("net save end")
    sys.exit()
