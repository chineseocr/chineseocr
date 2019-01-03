from config import yoloCfg,yoloWeights,opencvFlag
from config import AngleModelPb,AngleModelPbtxt
from config import IMGSIZE
from PIL import Image
import numpy as np
import cv2
textNet  = cv2.dnn.readNetFromDarknet(yoloCfg,yoloWeights)
if opencvFlag=='keras':
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

def text_detect(img):
    thresh=0
    h,w = img.shape[:2]
    inputBlob = cv2.dnn.blobFromImage(img, scalefactor=0.00390625, size=IMGSIZE,swapRB=True ,crop=False);
    textNet.setInput(inputBlob)
    pred = textNet.forward()
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
    if opencvFlag=='keras':
        return angle_detect_tf(img,adjust=adjust)
    else:
        return angle_detect_dnn(img,adjust=adjust)