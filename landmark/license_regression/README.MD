## 关键点网络训练说明
>> 1. 训练rgb图像和灰度图像的差异之处
a. example/train.py: train_rgb = True/False
b. network/backbone/basic_network.py: layer0 输入通道改为3/1

>> 2. 如何进行训练：
python example/train.py

>> 3.如何进行测试：
python example/test.py
