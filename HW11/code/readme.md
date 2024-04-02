211250074 左皓升

### 1. 运行配置说明

使用autodl服务器

GPU：RTX3090

OS：Ubuntu 20.04

CUDA：11.3

Pytorch：1.10.1

### 2. 文件结构说明

wandb文件夹下有wandb网站截图和报告

dragon文件夹为代码

train文件夹下为四个train任务的全部输出文件，包括各种log文件，控制台全部输出，结果json文件，wandb网站截图

eva问价夹下为检验作者模型运行的控制台全部截图

### 3. 数据集及模型地址

基础模型地址：

[biomed_model](https://nlp.stanford.edu/projects/myasu/DRAGON/models/biomed_model.pt)

[general_model](https://nlp.stanford.edu/projects/myasu/DRAGON/models/general_model.pt)

数据集地址：

[data](https://nlp.stanford.edu/projects/myasu/DRAGON/data_preprocessed.zip)

训练好的模型地址：

[[csqa\]](https://nlp.stanford.edu/projects/myasu/DRAGON/models/csqa_model.pt)

[[obqa\]](https://nlp.stanford.edu/projects/myasu/DRAGON/models/obqa_model.pt)

[[riddle\]](https://nlp.stanford.edu/projects/myasu/DRAGON/models/riddle_model.pt)

[[medqa\]](https://nlp.stanford.edu/projects/myasu/DRAGON/models/medqa_model.pt)

### 4. 其他说明

#### 4.1 bash

作者提供的代码中的bash文件是windows格式，在运行时先转化为unix格式才可以使用。由于运行是在云服务器上进行，这里提交的代码中bash文件未进行转化。

#### 4.2 huggingface

训练过程中需要连接huggingface获取模型，由于某些原因连接不到huggingface。于是在运行中将huggingface上的模型下载到本地再上传到服务器上，代码中路径更改为本地。这里提交的代码也没有进行更改，仍然为huggingface的路径。