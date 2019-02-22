"""Sanity tests for tf.flags."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from profile.angle_rotation import angle_detect, load_model
from tensorflow.python.platform import flags
from PIL import Image
import numpy as np
import traceback
import os

flags.DEFINE_string(name='image_dir', default='test', help='docstring')
flags.DEFINE_string(name='output_dir', default='test_out',
                    help='docstring')
flags.DEFINE_string(name='path', default=None, help='docstring')

FLAGS = flags.FLAGS


def folder_tester():
    try:
        print(FLAGS.output_dir)
        image_dir = os.listdir(FLAGS.image_dir)
        for pic in image_dir:
            single_pic = os.path.join(FLAGS.image_dir, pic)
            pic_flag = True if pic.endswith(".jpg") or pic.endswith(".png") or pic.endswith(".jpeg") else False
            if os.path.isfile(single_pic) and pic_flag:
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


def transpose_img(pic):
    img = Image.open(pic).convert("RGB")
    im = np.array(img)
    im_cp = np.copy(im)
    # load model
    load_model()
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


def main():
    folder_tester()


if __name__ == '__main__':
    main()
