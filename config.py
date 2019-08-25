import os
pwd = os.getcwd()
########################文字检测################################################
##文字检测引擎 
IMGSIZE = (608,608)## yolo3 输入图像尺寸
yoloTextFlag = 'keras' ##keras,opencv,darknet，模型性能 keras>darknet>opencv
############## keras yolo  ##############
keras_anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
class_names = ['none','text',]
kerasTextModel=os.path.join(pwd,"models","text.h5")##keras版本模型权重文件
############## keras yolo  ##############


############## darknet yolo  ##############

darknetRoot = os.path.join(os.path.curdir,"darknet")## yolo 安装目录
yoloCfg     = os.path.join(pwd,"models","text.cfg")
yoloWeights = os.path.join(pwd,"models","text.weights")
yoloData    = os.path.join(pwd,"models","text.data")
############## darknet yolo  ##############

########################文字检测################################################

## GPU选择及启动GPU序号
GPU = True##OCR 是否启用GPU
GPUID=0##调用GPU序号

##vgg文字方向检测模型
DETECTANGLE=True##是否进行文字方向检测
AngleModelPb    = os.path.join(pwd,"models","Angle-model.pb")
AngleModelPbtxt = os.path.join(pwd,"models","Angle-model.pbtxt")
AngleModelFlag  = 'opencv'  ## opencv or tf

######################OCR模型###################################################
ocr_redis = False##是否多任务执行OCR识别加速 如果多任务，则配置redis数据库，数据库账号参考apphelper/redisbase.py
##是否启用LSTM crnn模型
##OCR模型是否调用LSTM层
LSTMFLAG = True
ocrFlag = 'torch'##ocr模型 支持 keras  torch opencv版本
##模型选择 True:中英文模型 False:英文模型
chineseModel = True## 中文模型或者纯英文模型
##转换keras模型 参考tools目录
ocrModelKerasDense       = os.path.join(pwd,"models","ocr-dense.h5")
ocrModelKerasLstm        = os.path.join(pwd,"models","ocr-lstm.h5")
ocrModelKerasEng         = os.path.join(pwd,"models","ocr-english.h5")

ocrModelTorchLstm        = os.path.join(pwd,"models","ocr-lstm.pth")
ocrModelTorchDense       = os.path.join(pwd,"models","ocr-dense.pth")
ocrModelTorchEng         = os.path.join(pwd,"models","ocr-english.pth")

ocrModelOpencv           = os.path.join(pwd,"models","ocr.pb")

######################OCR模型###################################################

TIMEOUT=30##超时时间
