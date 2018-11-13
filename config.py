import os
opencvFlag = True##opencvFlag==True 启用opencv dnn 反之 darkent 
darknetRoot = os.path.join(os.path.curdir,"darknet")## yolo 安装目录
pwd = os.getcwd()
yoloCfg = os.path.join(pwd,"models","text.cfg")
yoloWeights = os.path.join(pwd,"models","text.weights")
yoloData = os.path.join(pwd,"models","text.data")
##文字方向检测
AngleModelPb = os.path.join(pwd,"models","Angle-model.pb")
AngleModelPbtxt = os.path.join(pwd,"models","Angle-model.pbtxt")
IMGSIZE = (1024,1024)## yolo3 输入图像尺寸
##是否启用LSTM crnn模型
DETECTANGLE=True##是否进行文字方向检测
LSTMFLAG = True##OCR模型是否调用LSTM层
GPU = True##OCR 是否启用GPU
GPUID=0##调用GPU序号
chinsesModel = True##模型选择 True:中英文模型 False:英文模型

if chinsesModel:
    if LSTMFLAG:
        ocrModel  = os.path.join(pwd,"models","ocr-lstm.pth")
    else:
        ocrModel = os.path.join(pwd,"models","ocr-dense.pth")
else:
        LSTMFLAG=True
        ocrModel = os.path.join(pwd,"models","ocr-english.pth")
