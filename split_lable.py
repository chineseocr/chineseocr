import os
import numpy as np
import math
import cv2 as cv

path = '/text-detection-ctpn/data/mlt_english+chinese/image'
gt_path = '/text-detection-ctpn/data/mlt_english+chinese/label'
out_path = 're_image'
if not os.path.exists(out_path):
    os.makedirs(out_path)
files = os.listdir(path)
files.sort()
#files=files[:100]
for file in files:
    _, basename = os.path.split(file)
    if basename.lower().split('.')[-1] not in ['jpg', 'png']:
        continue
    stem, ext = os.path.splitext(basename)
    gt_file = os.path.join(gt_path, 'gt_' + stem + '.txt')
    img_path = os.path.join(path, file)
    print(img_path)
    img = cv.imread(img_path)
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    re_im = cv.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv.INTER_LINEAR)
    re_size = re_im.shape
    cv.imwrite(os.path.join(out_path, stem) + '.jpg', re_im)

    with open(gt_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        splitted_line = line.strip().lower().split(',')
        pt_x = np.zeros((4, 1))
        pt_y = np.zeros((4, 1))
        pt_x[0, 0] = int(float(splitted_line[0]) / img_size[1] * re_size[1])
        pt_y[0, 0] = int(float(splitted_line[1]) / img_size[0] * re_size[0])
        pt_x[1, 0] = int(float(splitted_line[2]) / img_size[1] * re_size[1])
        pt_y[1, 0] = int(float(splitted_line[3]) / img_size[0] * re_size[0])
        pt_x[2, 0] = int(float(splitted_line[4]) / img_size[1] * re_size[1])
        pt_y[2, 0] = int(float(splitted_line[5]) / img_size[0] * re_size[0])
        pt_x[3, 0] = int(float(splitted_line[6]) / img_size[1] * re_size[1])
        pt_y[3, 0] = int(float(splitted_line[7]) / img_size[0] * re_size[0])

        ind_x = np.argsort(pt_x, axis=0)
        pt_x = pt_x[ind_x]
        pt_y = pt_y[ind_x]

        if pt_y[0] < pt_y[1]:
            pt1 = (pt_x[0], pt_y[0])
            pt3 = (pt_x[1], pt_y[1])
        else:
            pt1 = (pt_x[1], pt_y[1])
            pt3 = (pt_x[0], pt_y[0])

        if pt_y[2] < pt_y[3]:
            pt2 = (pt_x[2], pt_y[2])
            pt4 = (pt_x[3], pt_y[3])
        else:
            pt2 = (pt_x[3], pt_y[3])
            pt4 = (pt_x[2], pt_y[2])

        xmin = int(min(pt1[0], pt2[0]))
        ymin = int(min(pt1[1], pt2[1]))
        xmax = int(max(pt2[0], pt4[0]))
        ymax = int(max(pt3[1], pt4[1]))

        if xmin < 0:
            xmin = 0
        if xmax > re_size[1] - 1:
            xmax = re_size[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax > re_size[0] - 1:
            ymax = re_size[0] - 1

        width = xmax - xmin
        height = ymax - ymin

        # reimplement
        step = 16.0
        x_left = []
        x_right = []
        x_left.append(xmin)
        x_left_start = int(math.ceil(xmin / 16.0) * 16.0)
        if x_left_start == xmin:
            x_left_start = xmin + 16
        for i in np.arange(x_left_start, xmax, 16):
            x_left.append(i)
        x_left = np.array(x_left)

        x_right.append(x_left_start - 1)
        for i in range(1, len(x_left) - 1):
            x_right.append(x_left[i] + 15)
        x_right.append(xmax)
        x_right = np.array(x_right)

        idx = np.where(x_left == x_right)
        x_left = np.delete(x_left, idx, axis=0)
        x_right = np.delete(x_right, idx, axis=0)

        if not os.path.exists('label_tmp'):
            os.makedirs('label_tmp')
        with open(os.path.join('label_tmp', stem) + '.txt', 'a') as f:
            for i in range(len(x_left)):
                f.writelines("text\t")
                f.writelines(str(int(x_left[i])))
                f.writelines("\t")
                f.writelines(str(int(ymin)))
                f.writelines("\t")
                f.writelines(str(int(x_right[i])))
                f.writelines("\t")
                f.writelines(str(int(ymax)))
                f.writelines("\n")
