#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 00:28:01 2019
replace gpu/python/cython nms with cv2.dnn.NMS 
@author: chineseocr
"""
import cv2
from apphelper.image import solve
 
def nms(boxes, scores, score_threshold=0.5, nms_threshold=0.3):
    def box_to_center(box):
        xmin,ymin,xmax,ymax = [round(float(x),4) for x in box]
        w = xmax-xmin
        h = ymax-ymin
        return [round(xmin,4),round(ymin,4),round(w,4),round(h,4)]
    
    newBoxes = [ box_to_center(box) for box in boxes]
    newscores = [ round(float(x),6) for x in scores]
    index = cv2.dnn.NMSBoxes(newBoxes, newscores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    index = index.reshape((-1,))
    return boxes[index],scores[index]


def rotate_nms(boxes, scores, score_threshold=0.5, nms_threshold=0.3):
    """
    boxes.append((center, (w,h), angle * 180.0 / math.pi))
    box:x1,y1,x2,y2,x3,y3,x4,y4
    """
    def rotate_box(box):
       angle,w,h,cx,cy =  solve(box)
       angle = round(angle,4)
       w = round(w,4)
       h = round(h,4)
       cx = round(cx,4)
       cy = round(cy,4)
       
       return ((cx,cy),(w,h),angle)
   
    newboxes =  [rotate_box(box) for box in boxes]
    newscores = [ round(float(x),6) for x in scores]
    index = cv2.dnn.NMSBoxesRotated(newboxes, newscores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    if len(index)>0:
       index = index.reshape((-1,))
       return boxes[index],scores[index]
    else:
       return [],[]