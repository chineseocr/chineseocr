from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deploy.config import opencv_flag, AngleModelPb, AngleModelPbtxt
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os


if opencv_flag == 'keras':
    from tensorflow.python.platform import gfile

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    with gfile.FastGFile(AngleModelPb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    inputImg = sess.graph.get_tensor_by_name('input_1:0')
    predictions = sess.graph.get_tensor_by_name('predictions/Softmax:0')
    keep_prob = tf.placeholder(tf.float32)
else:
    angleNet = cv2.dnn.readNetFromTensorflow(AngleModelPb, AngleModelPbtxt)


def angle_detect_dnn(img, adjust=True):
    """
    文字方向检测
    """
    h, w = img.shape[:2]
    ROTATE = [0, 90, 180, 270]
    if adjust:
        thesh = 0.05
        xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        img = img[ymin:ymax, xmin:xmax]  ## 剪切图片边缘

    input_blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1.0,
        size=(224, 224),
        swapRB=True,
        mean=[103.939, 116.779, 123.68], crop=False)
    angleNet.setInput(input_blob)
    pred = angleNet.forward()
    index = np.argmax(pred, axis=1)[0]
    return ROTATE[index]


def angle_detect_tf(img, adjust=True):
    """
    文字方向检测
    """
    h, w = img.shape[:2]
    ROTATE = [0, 90, 180, 270]
    if adjust:
        thesh = 0.05
        xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        img = img[ymin:ymax, xmin:xmax]  ##剪切图片边缘
    img = cv2.resize(img, (224, 224))
    img = img[..., ::-1].astype(np.float32)

    img[..., 0] -= 103.939
    img[..., 1] -= 116.779
    img[..., 2] -= 123.68
    img = np.array([img])

    out = sess.run(predictions, feed_dict={inputImg: img, keep_prob: 0})

    index = np.argmax(out, axis=1)[0]
    return ROTATE[index]


# @profile
def angle_detect(img, adjust=True):
    """
    文字方向检测
    """
    if opencv_flag == 'keras':
        return angle_detect_tf(img, adjust=adjust)
    else:
        return angle_detect_dnn(img, adjust=adjust)


def main():
    bp = os.getcwd()
    idcard_pic = 'test/card.png'
    img = Image.open(idcard_pic).convert("RGB")
    w, h = img.size
    print("{} {} ".format(w, h))
    img_arr = np.array(img)
    img_cp = np.copy(img_arr)
    angle = angle_detect(img_cp, False)
    print("Angel : {}".format(angle))


if __name__ == "__main__":
    main()
