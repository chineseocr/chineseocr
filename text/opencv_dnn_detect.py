#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from config import yoloCfg,yoloWeights,AngleModelFlag
from config import AngleModelPb,AngleModelPbtxt
import numpy as np
import cv2
from apphelper.image import letterbox_image

if AngleModelFlag=='tf':
    ##转换为tf模型，以便GPU调用
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    with gfile.FastGFile(AngleModelPb, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
    inputImg =  sess.graph.get_tensor_by_name('input_1:0')
    predictions = sess.graph.get_tensor_by_name('predictions/Softmax:0')
    keep_prob = tf.placeholder(tf.float32)
    
else:
   angleNet = cv2.dnn.readNetFromTensorflow(AngleModelPb,AngleModelPbtxt)##dnn 文字方向检测
textNet  = cv2.dnn.readNetFromDarknet(yoloCfg,yoloWeights)##文字定位

def text_detect(img,scale,maxScale,prob = 0.05):
    thresh = prob

    img_height,img_width = img.shape[:2]
    inputBlob,f = letterbox_image(img,(scale,scale))
    inputBlob = cv2.dnn.blobFromImage(inputBlob, scalefactor=1.0, size=(scale,scale),swapRB=True ,crop=False);
    textNet.setInput(inputBlob/255.0)
    outputName = textNet.getUnconnectedOutLayersNames()
    outputs = textNet.forward(outputName)
    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > thresh:
                    center_x = int(detection[0] * scale/f)
                    center_y = int(detection[1] * scale/f)
                    width = int(detection[2] * scale/f)
                    height = int(detection[3] * scale/f)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    if class_id==1:
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([left, top,left+width, top+height ])
        
    boxes = np.array(boxes)
    confidences = np.array(confidences)
    
    return boxes,confidences



def angle_detect_dnn(img,adjust=True):
    """
    文字方向检测
    """
    h,w = img.shape[:2]
    ROTATE = [0,90,180,270]
    if adjust:
       thesh = 0.05
       xmin,ymin,xmax,ymax = int(thesh*w),int(thesh*h),w-int(thesh*w),h-int(thesh*h)
       img = img[ymin:ymax,xmin:xmax]##剪切图片边缘
    
    
    inputBlob = cv2.dnn.blobFromImage(img, 
                                      scalefactor=1.0, 
                                      size=(224, 224),
                                      swapRB=True ,
                                      mean=[103.939,116.779,123.68],crop=False);
    angleNet.setInput(inputBlob)
    pred = angleNet.forward()
    index = np.argmax(pred,axis=1)[0]
    return ROTATE[index]


def angle_detect_tf(img,adjust=True):
    """
    文字方向检测
    """
    h,w = img.shape[:2]
    ROTATE = [0,90,180,270]
    if adjust:
       thesh = 0.05
       xmin,ymin,xmax,ymax = int(thesh*w),int(thesh*h),w-int(thesh*w),h-int(thesh*h)
       img = img[ymin:ymax,xmin:xmax]##剪切图片边缘
    img = cv2.resize(img,(224,224))
    img = img[..., ::-1].astype(np.float32)
        
    img[..., 0] -= 103.939
    img[..., 1] -= 116.779
    img[..., 2] -= 123.68
    img          = np.array([img])
        
    
    
    out = sess.run(predictions, feed_dict={inputImg: img,
                                              keep_prob: 0
                                             })
    

    index = np.argmax(out,axis=1)[0]
    return ROTATE[index]

def angle_detect(img,adjust=True):
    """
    文字方向检测
    """
    if AngleModelFlag=='tf':
        return angle_detect_tf(img,adjust=adjust)
    else:
        return angle_detect_dnn(img,adjust=adjust)