#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
pwd = os.getcwd()

from config import yoloCfg,yoloWeights,yoloData,darknetRoot,GPU,GPUID
os.chdir(darknetRoot)
sys.path.append('python')
import darknet as dn


def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im

def detect_np(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = array_to_image(image)
    num = dn.c_int(0)
    pnum = dn.pointer(num)
    dn.predict_image(net, im)
    dets = dn.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): dn.do_nms_obj(dets, num, meta.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    dn.free_detections(dets, num)
    return res

def to_box(r):
    boxes = []
    scores = []
    for rc in r:
        if rc[0]==b'text':
            cx,cy,w,h = rc[-1]
            scores.append(rc[1])
            xmin,ymin,xmax,ymax = cx-w/2,cy-h/2,cx+w/2,cy+h/2
            boxes.append([int(xmin),int(ymin),int(xmax),int(ymax)])
    return boxes,scores


import pdb
if GPU:
    try:
      dn.set_gpu(GPUID)
    except:
        pass
    
net = dn.load_net(yoloCfg.encode('utf-8'), yoloWeights.encode('utf-8'), 0)
meta = dn.load_meta(yoloData.encode('utf-8'))
os.chdir(pwd)
def text_detect(img,scale,maxScale,prob = 0.05):
    
    r = detect_np(net, meta, img,thresh=prob, hier_thresh=0.5, nms=None)##输出所有box,与opencv dnn统一
    bboxes = to_box(r)
    return bboxes
