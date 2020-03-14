'''
@Author: Jiangtao
@Date: 2019-08-07 10:42:06
@LastEditors: Jiangtao
@LastEditTime: 2019-09-23 11:25:34
@Description: 
'''
import sys
sys.path.insert(0, '.')
sys.path.append("/home/mario/Projects/SSD/SSD_mobilenetv2/landmark/license_regression")
import engine


if __name__ == '__main__':

    samplePath5 = "/media/mario/新加卷/DataSets/ALPR/zhongdong/landmark/train.txt"
    model_Path = None
    samplePathList = []
    ## 设置训练采用rgb图像还是灰度图像，True表示为rgb图像，False表示灰度图像
    train_rgb = False
    inputSize=[]
    imgChannel = 1
    if train_rgb:
        inputSize=[128, 128, 3]
        imgChannel = 3
    else:
        inputSize=[128, 128, 1]
        imgChannel = 1

    samplePathList.append(samplePath5)

    eng = engine.TrainingEngine(modelPath=model_Path ,imgListPath=samplePathList, classNum=8, batchSize=128, workers=6, imgChannel = imgChannel, inputSize=inputSize)#
    eng()