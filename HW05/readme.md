### 运行说明

训练时将transformer也存储下来，测试时读取使用

train_model.py为训练代码，将33行路径替换为训练集路径，可以进行训练。由于模型较大，未放在压缩包中，models文件夹为空，保存在南大box和百度网盘上。

南大box链接：https://box.nju.edu.cn/f/89507f006ee745f8b138/?dl=1

百度网盘链接： https://pan.baidu.com/s/1ljzmfMGnIP_r1N0r1JOxWA?pwd=0000 提取码: 0000

1. 首先将box或百度网盘上models文件夹内容添加到本地，或者运行一次train_model.py（需要更改33行路径为训练集路径）

2. test_model.py为测试代码，将33行路径更改为测试集路径（压缩包中有，默认不需要更改），完成第一步后可以进行测试。

### 测试结果

测试结果会保存在preditions.json文件中，在结尾有正确的数量和准确率。经过本地多次测试，正确数量为469个，准确率为75.76%

![image-20240124015403745](https://haosn.oss-cn-beijing.aliyuncs.com/typora/image-20240124015403745.png)

![image-20240124015443549](https://haosn.oss-cn-beijing.aliyuncs.com/typora/image-20240124015443549.png)