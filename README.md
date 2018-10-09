## 本项目基于[yolo3](https://github.com/pjreddie/darknet.git) 与[crnn](https://github.com/meijieru/crnn.pytorch.git)  实现中文自然场景文字检测及识别

## 环境部署
python=3.6 pytorch==0.4.1
``` Bash
git clone https://github.com/chineseocr/chineseocr.git
cd chineseocr
sh setup.sh #(cpu sh setpu-cpu.sh)
```

下载编译darknet(如果直接运用opencv dnn 可忽略darknet的编译)
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
* [google drive](https://drive.google.com/drive/folders/1vlA6FjvicTt5GKvAfmycP5AlYxm4i9ze?usp=sharing)（暂时无更新）

复制文件夹中的所有文件到models目录

也可将yolo3模型转换为keras版本，详细参考https://github.com/qqwweee/keras-yolo3.git    

或者直接运用opencv>=3.4  dnn模块调用darknet模型(参考 opencv_dnn_detect.py)。   

## web服务启动
``` Bash
cd chineseocr## 进入chineseocr目录
ipython app.py 8080 ##8080端口号，可以设置任意端口
```

## 识别结果展示

<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/img1.png"/>
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/4.png"/>
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/card1.png"/>

## Play with Docker Container
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
5.https://github.com/qqwweee/keras-yolo3.git 

