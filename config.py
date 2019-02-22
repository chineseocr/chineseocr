import os

opencvFlag = 'keras'  ##keras,opencv,darknet
keras_anchors = '8,9, 8,18, 8,31, 8,59, 8,124, 8,351, 8,509, 8,605, 8,800'
class_names = ['none', 'text', ]
GPU = True  ##OCR 是否启用GPU
GPUID = 0  ##调用GPU序号

darknetRoot = os.path.join(os.path.curdir, "darknet")  ## yolo 安装目录
pwd = os.getcwd()
yoloCfg = os.path.join(pwd, "models", "text.cfg")
yoloWeights = os.path.join(pwd, "models", "text.weights")
yoloData = os.path.join(pwd, "models", "text.data")

kerasTextModel = os.path.join(pwd, "models", "text.h5")  ##keras版本模型

##文字方向检测
AngleModelPb = os.path.join(pwd, "models", "Angle-model.pb")
AngleModelPbtxt = os.path.join(pwd, "models", "Angle-model.pbtxt")
IMGSIZE = (608, 608)  ## yolo3 输入图像尺寸

##是否启用LSTM crnn模型
DETECTANGLE = True  ##是否进行文字方向检测
LSTMFLAG = True  ##OCR模型是否调用LSTM层
chinsesModel = True  ##模型选择 True:中英文模型 False:英文模型

if chinsesModel:
    if LSTMFLAG:
        ocrModel = os.path.join(pwd, "models", "ocr-lstm.pth")
    else:
        ocrModel = os.path.join(pwd, "models", "ocr-dense.pth")
else:
    LSTMFLAG = True
    ocrModel = os.path.join(pwd, "models", "ocr-english.pth")
