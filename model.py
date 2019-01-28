# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np
from PIL import Image
from glob import glob
from crnn.crnn import crnnOcr as crnnOcr 
from text.detector.detectors import TextDetector
from apphelper.image import get_boxes,letterbox_image
from config import opencvFlag,GPU,IMGSIZE
from text.opencv_dnn_detect import angle_detect##文字方向检测,支持dnn/tensorflow
from apphelper.image import estimate_skew_angle ,rotate_cut_img,xy_rotate_box,sort_box,box_rotate,solve

if opencvFlag=='opencv':
    from text import opencv_dnn_detect as detect ##opencv dnn model for darknet
elif opencvFlag=='darknet':
    from text import darknet_detect as detect
else:
    ## keras版本文字检测
    from text import keras_detect as detect
    

def text_detect(img,
                MAX_HORIZONTAL_GAP=30,
                MIN_V_OVERLAPS=0.6,
                MIN_SIZE_SIM=0.6,
                TEXT_PROPOSALS_MIN_SCORE=0.7,
                TEXT_PROPOSALS_NMS_THRESH=0.3,
                TEXT_LINE_NMS_THRESH = 0.3,
                MIN_RATIO=1.0,
                LINE_MIN_SCORE=0.8,
                TEXT_PROPOSALS_WIDTH=5,
                MIN_NUM_PROPOSALS=1,
                ):
    boxes, scores = detect.text_detect(np.array(img))

        
    boxes = np.array(boxes,dtype=np.float32)
    scores = np.array(scores,dtype=np.float32)
    textdetector = TextDetector(MAX_HORIZONTAL_GAP,MIN_V_OVERLAPS,MIN_SIZE_SIM)
    shape = img.size[::-1]
    boxes = textdetector.detect(boxes,
                                scores[:, np.newaxis],
                                shape,
                                TEXT_PROPOSALS_MIN_SCORE,
                                TEXT_PROPOSALS_NMS_THRESH,
                                TEXT_LINE_NMS_THRESH,
                                MIN_RATIO,
                                LINE_MIN_SCORE,
                                TEXT_PROPOSALS_WIDTH,
                                MIN_NUM_PROPOSALS)
    
    text_recs = get_boxes(boxes)
    newBox = []
    rx = 1
    ry = 1
    for box in text_recs:
           x1,y1 = (box[0],box[1])
           x2,y2 = (box[2],box[3])
           x3,y3 = (box[6],box[7])
           x4,y4 = (box[4],box[5])
           newBox.append([x1*rx,y1*ry,x2*rx,y2*ry,x3*rx,y3*ry,x4*rx,y4*ry])
    return newBox 



def crnnRec(im,boxes,leftAdjust=False,rightAdjust=False,alph=0.2,f=1.0):
   """
   crnn模型，ocr识别
   @@model,
   @@converter,
   @@im:Array
   @@text_recs:text box
   @@ifIm:是否输出box对应的img
   
   """
   results = []
   im = Image.fromarray(im) 
   for index,box in enumerate(boxes):

       degree,w,h,cx,cy = solve(box)
       partImg,newW,newH = rotate_cut_img(im,degree,box,w,h,leftAdjust,rightAdjust,alph)
       newBox = xy_rotate_box(cx,cy,newW,newH,degree)
       partImg_ = partImg.convert('L')
       simPred = crnnOcr(partImg_)##识别的文本
       if simPred.strip()!=u'':
            results.append({'cx':cx*f,'cy':cy*f,'text':simPred,'w':newW*f,'h':newH*f,'degree':degree*180.0/np.pi})
 
   return results




def eval_angle(im,detectAngle=False,ifadjustDegree=True):
    """
    估计图片偏移角度
    @@param:im
    @@param:ifadjustDegree 调整文字识别结果
    @@param:detectAngle 是否检测文字朝向
    """
    angle = 0
    degree=0.0
    img = np.array(im)
    if detectAngle:
        angle = angle_detect(img=np.copy(img))##文字朝向检测
        if angle==90:
            im = im.transpose(Image.ROTATE_90)
        elif angle==180:
            im = im.transpose(Image.ROTATE_180)
        elif angle==270:
            im = im.transpose(Image.ROTATE_270)
        img = np.array(im)
        
    if ifadjustDegree:
       degree = estimate_skew_angle(np.array(im.convert('L')))
    return  angle,degree,im.rotate(degree)


def model(img,detectAngle=False,config={},leftAdjust=False,rightAdjust=False,alph=0.2,ifadjustDegree=False):
    """
    @@param:img,
    @@param:ifadjustDegree 调整文字识别倾斜角度
    @@param:detectAngle,是否检测文字朝向
    """
    angle,degree,img = eval_angle(img,detectAngle=detectAngle,ifadjustDegree=ifadjustDegree)
    if opencvFlag!='keras':
       img,f =letterbox_image(img, IMGSIZE)
    else:
        f=1.0##解决box在原图坐标不一致问题
    
    config['img'] = img
    text_recs = text_detect(**config)
    
    newBox = sort_box(text_recs)
    result = crnnRec(np.array(img),newBox,leftAdjust,rightAdjust,alph,1.0/f)
    return img,result,angle



