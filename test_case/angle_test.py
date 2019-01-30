import time
from PIL import Image

p = './test/ocr3.jpeg'
img = Image.open(p).convert("RGB")
w, h = img.size
timeTake = time.time()

def eval_angle(im, detectAngle=False, ifadjustDegree=True):
    """
    估计图片偏移角度
    @@param:im
    @@param:ifadjustDegree 调整文字识别结果
    @@param:detectAngle 是否检测文字朝向
    """
    angle = 0
    degree = 0.0
    img = np.array(im)
    if detectAngle:
        angle = angle_detect(img=np.copy(img))  ##文字朝向检测
        if angle == 90:
            im = im.transpose(Image.ROTATE_90)
        elif angle == 180:
            im = im.transpose(Image.ROTATE_180)
        elif angle == 270:
            im = im.transpose(Image.ROTATE_270)
        img = np.array(im)

    if ifadjustDegree:
        degree = estimate_skew_angle(np.array(im.convert('L')))
    return angle, degree, im.rotate(degree)

angle, degree, img = eval_angle(img, detectAngle=detectAngle, ifadjustDegree=ifadjustDegree)
