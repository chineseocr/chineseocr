"""Sanity tests for tf.flags."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from deploy.angle_translate import angle_detect
from tensorflow.python.platform import flags
from PIL import Image
import numpy as np
import traceback
import os


flags.DEFINE_string(name='image_dir', default=None, help='docstring')
flags.DEFINE_string(name='output_dir', default=None, help='docstring')
flags.DEFINE_string(name='path', default=None, help='docstring')

FLAGS = flags.FLAGS


@profile
def folder_tester():
    try:
        print(FLAGS.output_dir)
        image_dir = os.listdir(FLAGS.image_dir)
        for pic in image_dir:
            single_pic = os.path.join(FLAGS.image_dir, pic)
            if os.path.isfile(single_pic) and pic.endswith(".jpg"):
                im = transpose_img(single_pic)
                if FLAGS.output_dir:
                    write_path = os.path.join(FLAGS.output_dir, pic)
                    print("... write to [{}] ...".format(write_path))
                    if isinstance(im, np.ndarray):
                        im = Image.fromarray(im)
                    im.save(write_path)
            else:
                print("... Jump pic {}...".format(single_pic))
    except BaseException as bxp:
        traceback.print_exc()
        print("Exception : {}".format(bxp))


@profile
def transpose_img(pic):
    img = Image.open(pic).convert("RGB")
    w, h = img.size
    im = np.array(img)
    im_cp = np.copy(im)
    angle = angle_detect(im_cp, False)
    print("Angel : [{}]".format(angle))

    if angle == 90:
        im = img.transpose(Image.ROTATE_90)
    elif angle == 180:
        im = img.transpose(Image.ROTATE_180)
    elif angle == 270:
        im = img.transpose(Image.ROTATE_270)

    # if ifadjustDegree:
    #     degree = estimate_skew_angle(np.array(im.convert('L')))
    # return angle, degree, im.rotate(degree)

    return im


if __name__ == '__main__':
    folder_tester()