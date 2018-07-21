import os
darknetRoot = os.path.join("..","darknet")## yolo 安装目录
pwd = os.getcwd()
yoloCfg = os.path.join(pwd,"models","text.cfg")
yoloWeights = os.path.join(pwd,"models","text.weights")
yoloData = os.path.join(pwd,"models","text.data")
ocrModel = os.path.join(pwd,"models","ocr.pth")
