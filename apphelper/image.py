# -*- coding: utf-8 -*-
"""
##图像相关函数
@author: lywen
"""
import sys
import six
import os
import base64
import requests
import numpy as np
import cv2
from PIL import Image
import traceback
import uuid
from glob import glob
from bs4 import BeautifulSoup
 
def sort_box_(box):
    x1,y1,x2,y2,x3,y3,x4,y4 = box[:8]
    pts = (x1,y1),(x2,y2),(x3,y3),(x4,y4)
    pts = np.array(pts, dtype="float32")
    (x1,y1),(x2,y2),(x3,y3),(x4,y4) = _order_points(pts)
    """
    newBox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    ## sort x
    newBox = sorted(newBox,key=lambda x:x[0])
    x1,y1 = sorted(newBox[:2],key=lambda x:x[1])[0]
    index = newBox.index([x1,y1])
    newBox.pop(index)
    newBox = sorted(newBox,key=lambda x:-x[1])
    x4,y4 = sorted(newBox[:2],key=lambda x:x[0])[0]
    index = newBox.index([x4,y4])
    newBox.pop(index)
    newBox = sorted(newBox,key=lambda x:-x[0])
    x2,y2 = sorted(newBox[:2],key=lambda x:x[1])[0]
    index = newBox.index([x2,y2])
    newBox.pop(index)
    
    newBox = sorted(newBox,key=lambda x:-x[1])
    x3,y3 = sorted(newBox[:2],key=lambda x:x[0])[0]
    """
    return x1,y1,x2,y2,x3,y3,x4,y4


import numpy as np
from scipy.spatial import distance as dist
def _order_points(pts):
    # 根据x坐标对点进行排序
    """
    --------------------- 
    作者：Tong_T 
    来源：CSDN 
    原文：https://blog.csdn.net/Tong_T/article/details/81907132 
    版权声明：本文为博主原创文章，转载请附上博文链接！
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # 从排序中获取最左侧和最右侧的点
    # x坐标点
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # 现在，根据它们的y坐标对最左边的坐标进行排序，这样我们就可以分别抓住左上角和左下角
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # 现在我们有了左上角坐标，用它作为锚来计算左上角和右上角之间的欧氏距离;
    # 根据毕达哥拉斯定理，距离最大的点将是我们的右下角
    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    # 返回左上角，右上角，右下角和左下角的坐标
    return np.array([tl, tr, br, bl], dtype="float32")



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
     if abs(sinA)>1:
            angle = None
     else:
        angle = np.arcsin(sinA)
     return angle,w,h,cx,cy

def read_singLine_for_yolo(p):
    """
    单行文本
    """
    im = Image.open(p).convert('RGB')
    w,h = im.size
    boxes = [{'cx':w/2,'cy':h/2,'w':w,'h':h,'angle':0.0}]
    return im,boxes
 
def read_voc_xml(p):
    ##读取voc xml 文件
    boxes = []
    if os.path.exists(p):
        with open(p) as f:
            xmlString = f.read()
        xmlString = BeautifulSoup(xmlString,'lxml')
        objList = xmlString.findAll('object')
        for obj in objList:
            robndbox = obj.find('robndbox')
            bndbox = obj.find('bndbox')
            if robndbox is not None and bndbox is None:
                cx = np.float(robndbox.find('cx').text)
                cy = np.float(robndbox.find('cy').text)
                w = np.float(robndbox.find('w').text)
                h = np.float(robndbox.find('h').text)
                angle = robndbox.find('angle').text
                if angle=='nan' or h==0 or w==0:
                    #boxes = []
                    continue
                    
                angle = np.float(angle)
                
                if abs(angle)>np.pi/2:
                    w,h = h,w
                    angle     = abs(angle)%(np.pi/2)*np.sign(angle)
                
                x1,y1,x2,y2,x3,y3,x4,y4 = xy_rotate_box(cx,cy,w,h,angle)
                x1,y1,x2,y2,x3,y3,x4,y4 = sort_box_([x1,y1,x2,y2,x3,y3,x4,y4])
                """
                if abs(angle)>np.pi/2:
                    ##lableImg bug
                    x1,y1,x2,y2,x3,y3,x4,y4 = sort_box_([x1,y1,x2,y2,x3,y3,x4,y4])
                """
                angle,w,h,cx,cy = solve([x1,y1,x2,y2,x3,y3,x4,y4])
                
            else:
                 xmin = np.float(bndbox.find('xmin').text)
                 xmax = np.float(bndbox.find('xmax').text)
                 ymin = np.float(bndbox.find('ymin').text)
                 ymax = np.float(bndbox.find('ymax').text)
                 cx = (xmin+xmax)/2.0
                 cy = (ymin+ymax)/2.0
                 w = (-xmin+xmax)#/2.0
                 h = (-ymin+ymax)#/2.0
                 angle =0.0
            boxes.append({'cx':cx,'cy':cy,'w':w,'h':h,'angle':angle})
                    
    return boxes


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
 
from numpy import cos,sin,pi,tan
def rotate(x,y,angle,cx,cy):
    """
    点(x,y) 绕(cx,cy)点旋转
    """
    #angle = angle*pi/180
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*cos(angle)+cy
    return x_new,y_new



def resize_box(boxes,scale):
    newBoxes = []
    for box in boxes:
        cx = box['cx']*scale
        cy = box['cy']*scale
        w  = box['w']*scale
        h  = box['h']*scale
        angle = box['angle']
        newBoxes.append({'cx':cx,'cy':cy,'w':w,'h':h,'angle':angle})
    return newBoxes
        
def resize_im(w,h, scale=416, max_scale=608):
    f=float(scale)/min(h, w)
    if max_scale is not None:
        if  f*max(h, w)>max_scale:
            f=float(max_scale)/max(h, w)
    newW,newH = int(w*f),int(h*f)
    
    return newW-(newW%32),newH-(newH%32)


def get_rorate(boxes,im,degree=0):
    """
    获取旋转角度后的box及im
    """
    imgW,imgH = im.size
    newBoxes = []       
    for line in boxes:
         cx0,cy0 = imgW/2.0,imgH/2.0 
         x1,y1,x2,y2,x3,y3,x4,y4 = xy_rotate_box(**line)
         x1,y1  = rotate(x1,y1,-degree/180*np.pi,cx0,cy0)
         x2,y2  = rotate(x2,y2,-degree/180*np.pi,cx0,cy0)
         x3,y3  = rotate(x3,y3,-degree/180*np.pi,cx0,cy0)
         x4,y4  = rotate(x4,y4,-degree/180*np.pi,cx0,cy0)
         box = (x1,y1,x2,y2,x3,y3,x4,y4)
         degree_,w_,h_,cx_,cy_ = solve(box)
         newLine = {'angle':degree_,'w':w_,'h':h_,'cx':cx_,'cy':cy_}
         newBoxes.append(newLine)
    return im.rotate(degree,center=(imgW/2.0,imgH/2.0 )),newBoxes


def letterbox_image(image, size,fillValue=[128,128,128]):
    '''
    resize image with unchanged aspect ratio using padding
    '''
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)
    if fillValue is None:
       fillValue = [int(x.mean()) for x in cv2.split(np.array(im))]
    boxed_image = Image.new('RGB', size, tuple(fillValue))
    
    boxed_image.paste(resized_image,)
    return boxed_image,new_w/image_w


def box_split(boxes,splitW = 15):
    newBoxes = []
    for box in boxes:
        w = box['w']
        h = box['h']
        cx = box['cx']
        cy=box['cy']
        angle = box['angle']
        x1,y1,x2,y2,x3,y3,x4,y4 = xy_rotate_box(cx,cy,w,h,angle)
        splitBoxes =[]
        i = 1
        tanAngle = tan(-angle)
        
        while True:
            flag = 0 if i==1 else 1
            xmin = x1+(i-1)*splitW
            ymin = y1-tanAngle*splitW*i
            xmax = x1+i*splitW
            ymax = y4-(i-1)*tanAngle*splitW +flag*tanAngle*(x4-x1)
            if xmax>max(x2,x3) and xmin>max(x2,x3):
                break
            splitBoxes.append([int(xmin),int(ymin),int(xmax),int(ymax)])
            
            i+=1
        
        newBoxes.append(splitBoxes)
    
    return newBoxes

def get_box_spilt(boxes,im,sizeW,SizeH,splitW=8,isRoate=False,rorateDegree=0):
    """
    isRoate:是否旋转box
    """
    size = sizeW,SizeH

    if isRoate:
        ##旋转box
        im,boxes = get_rorate(boxes,im,degree=rorateDegree)
        
    newIm,f  = letterbox_image(im, size)
    newBoxes = resize_box(boxes,f)
    newBoxes = sum(box_split(newBoxes,splitW),[])
    newBoxes = [box+[1] for box in newBoxes]
    return newBoxes,newIm



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

     sinA = (h*(x1-cx)-w*(y1 -cy))*1.0/(h*h+w*w)*2
     angle = np.arcsin(sinA)
     return angle,w,h,cx,cy
    
 
from numpy import cos,sin,pi
def rotate(x,y,angle,cx,cy):
    angle = angle#*pi/180
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*cos(angle)+cy
    return x_new,y_new
    
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



def letterbox_image(image, size,fillValue=[128,128,128]):
    '''resize image with unchanged aspect ratio using padding'''
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)
    if fillValue is None:
       fillValue = [int(x.mean()) for x in cv2.split(np.array(im))]
    boxed_image = Image.new('RGB', size, tuple(fillValue))
    boxed_image.paste(resized_image, (0,0))
    return boxed_image,new_w/image_w

from scipy.ndimage import filters,interpolation,morphology,measurements,minimum
#from pylab import amin, amax
from numpy import amin, amax
def estimate_skew_angle(raw):
    """
    估计图像文字角度
    """
    
    def resize_im(im, scale, max_scale=None):
        f=float(scale)/min(im.shape[0], im.shape[1])
        if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
            f=float(max_scale)/max(im.shape[0], im.shape[1])
        return cv2.resize(im, (0, 0), fx=f, fy=f)

    raw = resize_im(raw, scale=600, max_scale=900)
    image = raw-amin(raw)
    image = image/amax(image)
    m = interpolation.zoom(image,0.5)
    m = filters.percentile_filter(m,80,size=(20,2))
    m = filters.percentile_filter(m,80,size=(2,20))
    m = interpolation.zoom(m,1.0/0.5)

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


def get_boxes( bboxes):
    """
        boxes: bounding boxes
    """
    text_recs=np.zeros((len(bboxes), 8), np.int)
    index = 0
    for box in bboxes:
        
        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2
        
        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX*disX + disY*disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1*disX / width)
        y = np.fabs(fTmp1*disY / width)
        if box[5] < 0:
           x1 -= x
           y1 += y
           x4 += x
           y4 -= y
        else:
           x2 += x
           y2 += y
           x3 -= x
           y3 -= y

        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
        index = index + 1

    return text_recs



def union_rbox(result,alpha=0.1):
    """
    按行合并box
    """            
    def diff(box1,box2):
        """
        计算box1,box2之间的距离
        """
        cy1 = box1['cy']
        cy2 = box2['cy']
        h1  = box1['h']
        h2 = box2['h']
        
        return abs(cy1-cy2)/max(0.01,min(h1/2,h2/2))
    
    def sort_group_box(boxes):
        """
        对box进行排序, 并合并box
        """   
        N = len(boxes)
        boxes = sorted(boxes,key=lambda x:x['cx'])
        text  = ' '.join([bx['text'] for bx in boxes])
        box4 = np.zeros((N,8))
        for i in range(N):
            cx =boxes[i]['cx']
            cy = boxes[i]['cy']
            degree =boxes[i]['degree']
            w  = boxes[i]['w']
            h = boxes[i]['h']
            x1,y1,x2,y2,x3,y3,x4,y4 = xy_rotate_box(cx, cy, w, h, degree/180*np.pi)
            box4[i] = [x1,y1,x2,y2,x3,y3,x4,y4]
            
        x1 = box4[:,0].min()
        y1 = box4[:,1].min()
        x2 = box4[:,2].max()
        y2 = box4[:,3].min()
        x3 = box4[:,4].max()
        y3 = box4[:,5].max()
        x4 = box4[:,6].min()
        y4 = box4[:,7].max()
        angle,w,h,cx,cy = solve([x1,y1,x2,y2,x3,y3,x4,y4])
        return {'text':text,'cx':cx,'cy':cy,'w':w,'h':h,'degree':angle/np.pi*180}
    
    

    newBox = []
    for line in result:
        if len(newBox)==0:
            newBox.append([line])
        else:
            check=False
            for box in newBox[-1]:
                if diff(line,box)>alpha:
                    check = True
                    
            if not check:
                newBox[-1].append(line)
            else:
                newBox.append([line])
    newBox = [sort_group_box(bx) for bx in newBox]
    return newBox
            

def adjust_box_to_origin(img,angle, result):
    """
    调整box到原图坐标
    """
    h,w = img.shape[:2]
    if angle in [90,270]:
        imgW,imgH = img.shape[:2]
        
    else:
        imgH,imgW= img.shape[:2]
    newresult = []
    for line in result:
        cx =line['box']['cx']
        cy = line['box']['cy']
        degree =line['box']['angle']
        w  = line['box']['w']
        h = line['box']['h']
        x1,y1,x2,y2,x3,y3,x4,y4 = xy_rotate_box(cx, cy, w, h, degree/180*np.pi)
        x1,y1,x2,y2,x3,y3,x4,y4 = box_rotate([x1,y1,x2,y2,x3,y3,x4,y4],angle=(360-angle)%360,imgH=imgH,imgW=imgW)
        box = x1,y1,x2,y2,x3,y3,x4,y4
        newresult.append({'name':line['name'],'text':line['text'],'box':box})
       
    return newresult