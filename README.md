## 本项目基于[yolo3](https://github.com/pjreddie/darknet.git) 与[crnn](https://github.com/meijieru/crnn.pytorch.git)  实现中文自然场景文字检测及识别
master分支将保留一周，后续app分支将替换为master    

# 实现功能
- [x]  文字方向检测 0、90、180、270度检测（支持dnn/tensorflow） 
- [x]  支持(darknet/opencv dnn /keras)文字检测,支持darknet/keras训练
- [x]  不定长OCR训练(英文、中英文) crnn\dense ocr 识别及训练 ,新增pytorch转keras模型代码(tools/pytorch_to_keras.py)
- [x]  支持darknet 转keras, keras转darknet, pytorch 转keras模型
- [x]  身份证/火车票结构化数据识别 
- [x]  新增CNN+ctc模型，支持DNN模块调用OCR，单行图像平均时间为0.02秒以下     
- [ ]  CPU版本加速    
- [ ]  支持基于用户字典OCR识别    
- [ ]  新增语言模型修正OCR识别结果  
- [ ]  支持树莓派实时识别方案  
 

## 环境部署

GPU部署 参考:setup.md     
CPU部署 参考:setup-cpu.md   


### 下载编译darknet(如果直接运用opencv dnn或者keras yolo3 可忽略darknet的编译)  
```
git clone https://github.com/pjreddie/darknet.git 
mv darknet chineseocr/
##编译对GPU、cudnn的支持 修改 Makefile
#GPU=1
#CUDNN=1
#OPENCV=0
#OPENMP=0
make 
```

修改 darknet/python/darknet.py line 48    
root = '/root/'##chineseocr所在目录     
lib = CDLL(root+"chineseocr/darknet/libdarknet.so", RTLD_GLOBAL)    


## 下载模型文件   
模型文件地址:
* [baidu pan](https://pan.baidu.com/s/1gTW9gwJR6hlwTuyB6nCkzQ)

复制文件夹中的所有文件到models目录
   
## 模型转换
pytorch ocr 转keras ocr     
``` Bash
python tools/pytorch_to_keras.py  -weights_path models/ocr-dense.pth -output_path models/ocr-dense-keras.h5
```
darknet 转keras     
``` Bash
python tools/darknet_to_keras.py -cfg_path models/text.cfg -weights_path models/text.weights -output_path models/text.h5
```
keras 转darknet      
``` Bash
python tools/keras_to_darknet.py -cfg_path models/text.cfg -weights_path models/text.h5 -output_path models/text.weights
```

## 编译语言模型
``` Bash
git clone --recursive https://github.com/parlance/ctcdecode.git   
cd ctcdecode   
pip install .  
```
## 下载语言模型  
``` Bash
wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
mv zh_giga.no_cna_cmn.prune01244.klm chineseocr/models/
```
## 模型选择  
``` Bash
参考config.py文件
```  

## 构建docker镜像 
``` Bash
##下载Anaconda3 python 环境安装包（https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh） 放置在chineseocr目录下   
##建立镜像   
docker build -t chineseocr .   
##启动服务   
docker run
