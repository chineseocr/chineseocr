#! /usr/bin/env python
# -*- coding: utf-8 -*-
from detector.detectors import TextDetector
from detector.other import get_boxes
from config import opencvFlag
from config import IMGSIZE
from opencv_dnn_detect import angle_detect##文字方向检测
if opencvFlag:
    import opencv_dnn_detect as detect ##opencv dnn model for darknet
else:
    import darknet_detect as detect

import numpy as np
from PIL import Image
import numpy as np
import time
import cv2
from glob import glob
from crnn.crnn import crnnOcr as crnnOcr 

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
    textdetector  = TextDetector(MAX_HORIZONTAL_GAP,MIN_V_OVERLAPS,MIN_SIZE_SIM)
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
    
    text_recs,tmp = get_boxes(np.array(img), boxes)
    newBox = []
    rx = 1
    ry = 1
    for box in text_recs:
           x1,y1 = (box[0],box[1])
           x2,y2 = (box[2],box[3])
           x3,y3 = (box[6],box[7])
           x4,y4 = (box[4],box[5])
           newBox.append([x1*rx,y1*ry,x2*rx,y2*ry,x3*rx,y3*ry,x4*rx,y4*ry])
    return newBox,tmp 


import numpy as np
from PIL import Image
def crnnRec(im,boxes,ifIm=False,leftAdjust=False,rightAdjust=False,alph=0.2):
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
            results.append({'cx':cx,'cy':cy,'text':simPred,'w':newW,'h':newH,'degree':degree*180.0/np.pi})
 
   return results

def box_rotate(box,angle=0,imgH=0,imgW=0):
    """
    对坐标进行旋转 逆时针方向 0\90\180\270,
    """
    x1,y1,x2,y2,x3,y3,x4,y4 = box[:8]
    if angle==90:
        x1_,y1_ = y2,imgW-x2
        x2_,y2_ = y3,imgW-x3
        x3_,y3_ = y4,imgW-x4
        x4_,y4_ = y1,imgW-x1
        
    elif angle==180:
        x1_,y1_ = imgW-x3,imgH-y3
        x2_,y2_ = imgW-x4,imgH-y4
        x3_,y3_ = imgW-x1,imgH-y1
        x4_,y4_ = imgW-x2,imgH-y2
        
    elif angle==270:
        x1_,y1_ = imgH-y4,x4
        x2_,y2_ = imgH-y1,x1
        x3_,y3_ = imgH-y2,x2
        x4_,y4_ = imgH-y3,x3
    else:
        x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_  = x1,y1,x2,y2,x3,y3,x4,y4
        
    return (x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_)


def solve(box):
     """
     绕 cx,cy点 w,h 旋转 angle 的坐标
     x = cx-w/2
     y = cy-h/2
     x1-cx = -w/2*cos(angle) +h/2*sin(angle)
     y1 -cy= -w/2*sin(angle) -h/2*cos(angle)
     
     h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
     w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
     (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

     """
     x1,y1,x2,y2,x3,y3,x4,y4= box[:8]
     cx = (x1+x3+x2+x4)/4.0
     cy = (y1+y3+y4+y2)/4.0  
     w = (np.sqrt((x2-x1)**2+(y2-y1)**2)+np.sqrt((x3-x4)**2+(y3-y4)**2))/2
     h = (np.sqrt((x2-x3)**2+(y2-y3)**2)+np.sqrt((x1-x4)**2+(y1-y4)**2))/2   
     #x = cx-w/2
     #y = cy-h/2
     sinA = (h*(x1-cx)-w*(y1 -cy))*1.0/(h*h+w*w)*2
     angle = np.arcsin(sinA)
     return angle,w,h,cx,cy
    
def xy_rotate_box(cx,cy,w,h,angle):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*sin(angle)+cy
    """
        
    cx    = float(cx)
    cy    = float(cy)
    w     = float(w)
    h     = float(h)
    angle = float(angle)
    x1,y1 = rotate(cx-w/2,cy-h/2,angle,cx,cy)
    x2,y2 = rotate(cx+w/2,cy-h/2,angle,cx,cy)
    x3,y3 = rotate(cx+w/2,cy+h/2,angle,cx,cy)
    x4,y4 = rotate(cx-w/2,cy+h/2,angle,cx,cy)
    return x1,y1,x2,y2,x3,y3,x4,y4
 
from numpy import cos,sin,pi
def rotate(x,y,angle,cx,cy):
    angle = angle#*pi/180
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*cos(angle)+cy
    return x_new,y_new
                                
                                
def rotate_cut_img(im,degree,box,w,h,leftAdjust=False,rightAdjust=False,alph=0.2):
    x1,y1,x2,y2,x3,y3,x4,y4 = box[:8]
    x_center,y_center = np.mean([x1,x2,x3,x4]),np.mean([y1,y2,y3,y4])
    degree_ = degree*180.0/np.pi
    right = 0
    left  = 0
    if rightAdjust:
        right = 1
    if leftAdjust:
        left  = 1
    
    box = (max(1,x_center-w/2-left*alph*(w/2))##xmin
           ,y_center-h/2,##ymin
           min(x_center+w/2+right*alph*(w/2),im.size[0]-1)##xmax
           ,y_center+h/2)##ymax
 
    newW = box[2]-box[0]
    newH = box[3]-box[1]
    tmpImg = im.rotate(degree_,center=(x_center,y_center)).crop(box)
    return tmpImg,newW,newH


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding
        Reference: https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/utils.py
        '''
    image_w, image_h = image.size
    w, h = size
    
    if max(image_w, image_h)<min(size):
        resized_image = image
        new_w = w
        new_h = h
    else:
        new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
        new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
        resized_image = image.resize((new_w,new_h), Image.BICUBIC)
    
    boxed_image = Image.new('RGB', size, (128,128,128))
    boxed_image.paste(resized_image, ((w-new_w)//2,(h-new_h)//2))
    return boxed_image

from scipy.ndimage import filters,interpolation,morphology,measurements,minimum
#from pylab import amin, amax
from numpy import amin, amax
def estimate_skew_angle(raw):
    """
    估计图像文字角度
    """
    raw = resize_im(raw, scale=600, max_scale=900)
    image = raw-amin(raw)
    image = image/amax(image)
    m = interpolation.zoom(image,0.5)
    m = filters.percentile_filter(m,80,size=(20,2))
    m = filters.percentile_filter(m,80,size=(2,20))
    m = interpolation.zoom(m,1.0/0.5)
    #w,h = image.shape[1],image.shape[0]
    w,h = min(image.shape[1],m.shape[1]),min(image.shape[0],m.shape[0])
    flat = np.clip(image[:h,:w]-m[:h,:w]+1,0,1)
    d0,d1 = flat.shape
    o0,o1 = int(0.1*d0),int(0.1*d1)
    flat = amax(flat)-flat
    flat -= amin(flat)
    est = flat[o0:d0-o0,o1:d1-o1]
    angles = range(-15,15)
    estimates = []
    for a in angles:
        
        roest =interpolation.rotate(est,a,order=0,mode='constant')
        v = np.mean(roest,axis=1)
        v = np.var(v)
        estimates.append((v,a))
    
    _,a = max(estimates)
    return a


def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, (0, 0), fx=f, fy=f)


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


def model(img,detectAngle=False,config={},ifIm=True,leftAdjust=False,rightAdjust=False,alph=0.2,ifadjustDegree=False):
    """
    @@param:img,
    @@param:adjust 调整文字识别结果
    @@param:detectAngle,是否检测文字朝向
    """
    angle,degree,img = eval_angle(img,detectAngle=detectAngle,ifadjustDegree=ifadjustDegree)
    
    img =letterbox_image(img, IMGSIZE)
    
    config['img'] = img
    text_recs,tmp = text_detect(**config)
    
    newBox = sort_box(text_recs)
    result = crnnRec(np.array(img),newBox,ifIm,leftAdjust,rightAdjust,alph)
    return img,result,angle

def sort_box(box):
    """
    对box排序,及页面进行排版
        box[index, 0] = x1
        box[index, 1] = y1
        box[index, 2] = x2
        box[index, 3] = y2
        box[index, 4] = x3
        box[index, 5] = y3
        box[index, 6] = x4
        box[index, 7] = y4
    """
    
    box = sorted(box,key=lambda x:sum([x[1],x[3],x[5],x[7]]))
    return list(box)

