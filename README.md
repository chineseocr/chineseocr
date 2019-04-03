## 本项目基于[yolo3](https://github.com/pjreddie/darknet.git) 与[crnn](https://github.com/meijieru/crnn.pytorch.git)  实现中文自然场景文字检测及识别

# 实现功能
- [x]  文字方向检测 0、90、180、270度检测（支持dnn/tensorflow） 
- [x]  支持(darknet/opencv dnn /keras)文字检测,支持darknet/keras训练
- [x]  不定长OCR训练(英文、中英文) crnn\dense ocr 识别及训练 ,新增pytorch转keras模型代码(tools/pytorch_to_keras.py)
- [x]  新增对身份证/火车票结构化数据识别

## 环境部署

GPU部署 参考:setup.md     
GPU部署 参考:setup-cpu.md   


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
* [google drive](https://drive.google.com/drive/folders/1XiT1FLFvokAdwfE9WSUSS1PnZA34WBzy?usp=sharing)

复制文件夹中的所有文件到models目录
   

## web服务启动
``` Bash
cd chineseocr## 进入chineseocr目录
ipython app.py 8080 ##8080端口号，可以设置任意端口
```

## 识别结果展示

<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/train-demo.png"/>
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/idcard-demo.png"/>
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/img-demo.png"/>
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/line-demo.png"/>

## Play with Docker Container(镜像有些滞后)
``` Bash
docker pull zergmk2/chineseocr
docker run -d -p 8080:8080 zergmk2/chineseocr
```

## 访问服务
http://127.0.0.1:8080/ocr

<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/demo.png"/>


## 参考
1. yolo3 https://github.com/pjreddie/darknet.git   
2. crnn  https://github.com/meijieru/crnn.pytorch.git              
3. ctpn  https://github.com/eragonruan/text-detection-ctpn    
4. CTPN  https://github.com/tianzhi0549/CTPN       
5. keras yolo3 https://github.com/qqwweee/keras-yolo3.git 

