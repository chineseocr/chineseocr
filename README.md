## 本项目基于[yolo3](https://github.com/pjreddie/darknet.git) 与[crnn](https://github.com/meijieru/crnn.pytorch.git)  实现中文自然场景文字检测及识别

## 环境部署
python=3.6 pytorch=0.2.0
``` Bash
git clone https://github.com/chineseocr/chineseocr.git
cd chineseocr
sh setup.sh #(cpu sh setpu-cpu.sh)
```

## 下载模型文件   
模型文件地址:
* [baidu pan](https://pan.baidu.com/s/1dEopm8VkFLuDDfVkpuzHqQ)
* [google drive](https://drive.google.com/drive/folders/1vlA6FjvicTt5GKvAfmycP5AlYxm4i9ze?usp=sharing)

复制文件夹中的 ocr.pth, text.cfg, text.names, text.weights 到models目录

或者可将yolo3模型转换为keras版本，详细参考https://github.com/qqwweee/keras-yolo3.git    

或者直接运用opencv>=3.4  dnn直接调用darknet模型（参考 opencv_dnn_detect.py），cpu文本检测小于1秒。   

## web服务启动
``` Bash
cd chineseocr## 进入chineseocr目录
ipython app.py 8080 ##8080端口号，可以设置任意端口
```

## 识别结果展示

<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/img1.png"/>
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/4.png"/>

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

