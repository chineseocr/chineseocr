import cv2
import numpy as np


def prepare_img(im, mean):
    """
        transform img into caffe's input img.
    """
    im_data=np.transpose(im-mean, (2, 0, 1))
    return im_data


def get_boxes(im, bboxes):
    """
        boxes: bounding boxes
	"""
    text_recs=np.zeros((len(bboxes), 8), np.int)
    im=im.copy()
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

    return text_recs,im


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, im_shape[1]-1)
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, im_shape[0]-1)
    return boxes


def normalize(data):
    if data.shape[0]==0:
        return data
    max_=data.max()
    min_=data.min()
    return (data-min_)/(max_-min_) if max_-min_!=0 else data-min_


def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, (0, 0), fx=f, fy=f), f
    #return cv2.resize(im, (0, 0), fx=1.2, fy=1.2), f





