from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os


class Config:
    opencv_flag = 'keras'
    pwd = os.path.split(__file__)[0]
    AngleModelPb = os.path.join(pwd, "models", "Angle-model.pb")
    AngleModelPbtxt = os.path.join(pwd, "models", "Angle-model.pbtxt")


class Assets:
    basemodel = None
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    input_placeholder = None
    predictions = None
    keep_prob = None


def load_model(opencv_flag='keras'):
    if opencv_flag == 'keras':
        from tensorflow.python.platform import gfile
        with gfile.FastGFile(Config.AngleModelPb, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            Assets.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        Assets.input_placeholder = Assets.sess.graph.get_tensor_by_name('input_1:0')
        Assets.predictions = Assets.sess.graph.get_tensor_by_name('predictions/Softmax:0')
        Assets.keep_prob = tf.placeholder(tf.float32)
    else:
        Assets.basemodel = cv2.dnn.readNetFromTensorflow(Config.AngleModelPb, Config.AngleModelPbtxt)


def angle_detect_dnn(img, adjust=True):
    """
    文字方向检测
    """
    h, w = img.shape[:2]
    ROTATE = [0, 90, 180, 270]
    if adjust:
        thesh = 0.05
        xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        img = img[ymin:ymax, xmin:xmax]

    input_blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1.0,
        size=(224, 224),
        swapRB=True,
        mean=[103.939, 116.779, 123.68], crop=False)
    Assets.basemodel.setInput(input_blob)
    pred = Assets.basemodel.forward()
    index = np.argmax(pred, axis=1)[0]
    return ROTATE[index]


# @profile
def angle_detect_tf(img, adjust=True):
    """
    文字方向检测
    """
    h, w = img.shape[:2]
    ROTATE = [0, 90, 180, 270]
    if adjust:
        thesh = 0.05
        xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        img = img[ymin:ymax, xmin:xmax]
    img = cv2.resize(img, (224, 224))
    img = img[..., ::-1].astype(np.float32)

    img[..., 0] -= 103.939
    img[..., 1] -= 116.779
    img[..., 2] -= 123.68
    m_img = np.array([img])

    out = Assets.sess.run(Assets.predictions, feed_dict={Assets.input_placeholder: m_img, Assets.keep_prob: 0})

    index = np.argmax(out, axis=1)[0]
    print("index:{} ".format(index))
    print('ROTATE[index] : {}'.format(ROTATE[index]))
    return ROTATE[index]


# @profile
def angle_detect(img, adjust=True, opencv_flag='keras'):
    """
    文字方向检测
    """
    if opencv_flag == 'keras':
        return angle_detect_tf(img, adjust=adjust)
    else:
        return angle_detect_dnn(img, adjust=adjust)


def main():
    base_path = os.path.dirname(os.path.split(__file__)[0])
    print("base path: {}".format(base_path))
    load_model()
    idcard_pic = os.path.join(base_path, 'test/card.png')
    img = Image.open(idcard_pic).convert("RGB")
    w, h = img.size
    print("{} {} ".format(w, h))
    img_arr = np.array(img)
    img_cp = np.copy(img_arr)
    angle = angle_detect(img_cp, False)
    print("Angel : {}".format(angle))


if __name__ == "__main__":
    main()
