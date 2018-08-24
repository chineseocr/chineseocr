from config import yoloCfg,yoloWeights
from PIL import Image
import numpy as np
import cv2
net = cv2.dnn.readNetFromDarknet(yoloCfg,yoloWeights)
def text_detect(img):
    thresh=0.1
    h,w = img.shape[:2]
    inputBlob = cv2.dnn.blobFromImage(img, scalefactor=0.00390625, size=(608, 608),swapRB=True ,crop=False);
    net.setInput(inputBlob)
    pred = net.forward()
    cx = pred[:,0]*w
    cy = pred[:,1]*h
    xmin = cx - pred[:,2]*w/2
    xmax = cx + pred[:,2]*w/2
    ymin = cy - pred[:,3]*h/2
    ymax = cy + pred[:,3]*h/2
    scores = pred[:,4]
    indx = np.where(scores>thresh)[0]
    scores = scores[indx]
    boxes = np.array(list(zip(xmin[indx],ymin[indx],xmax[indx],ymax[indx])))
    return boxes,scores
