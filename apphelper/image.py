# -*- coding: utf-8 -*-
"""
##图像相关函数
@author: lywen
"""
import base64
#import urllib2
import requests
import numpy as np
import cv2
from PIL import Image
import sys
import six
import traceback
import uuid

    
    
from PIL import Image   
def read_img(path):
    im = Image.open(path).convert('RGB')
    img = np.array(im)
    return img 
  
def convert_image(path=None,string=None):  
    # Picture ==> base64 encode 
    if path is not None:
        with open(path, 'rb') as f:  
            base64_data = base64.b64encode(f.read())
        return base64_data
    if string is not None:
        base64_data = base64.b64encode(string)
        return base64_data
    
def string_to_array(string):
    if check_image_is_valid(string):
                buf = six.BytesIO()
                buf.write(string)
                buf.seek(0)
                img = np.array(Image.open(buf))
                return img
    else:
            return None

def check_image_is_valid(imageBin):
    """
    检查图片是否有效
    """
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def base64_to_array(string):
    #try:
            base64_data = base64.b64decode(string)
            buf = six.BytesIO()
            buf.write(string)
            buf.seek(0)
            img = np.array(Image.open(buf))
            return img
        
    #except:
     #   return None
    
            

def read_url_img(url):
    """
    爬取网页图片
    """
    try:
        req = requests.get(url,timeout=5)##访问时间超过5s，则超时
        if req.status_code==200:
            imgString = req.content
        #imgString = req.read()
        if check_image_is_valid(imgString):
                buf = six.BytesIO()
                buf.write(imgString)
                buf.seek(0)
                img = Image.open(buf).convert('RGB')
                return img
        else:
            return None
    except:
        traceback.print_exc()
        return None
     

def array_to_string(array):
    image = Image.fromarray(array)
    output = six.BytesIO()
    image.save(output,format='png')
    contents = output.getvalue()
    output.close()
    return contents
