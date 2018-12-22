# -*- coding: utf-8 -*-
"""
@author: lywen
"""
import os
from PIL import Image
import json
import time
import uuid
import base64
import web
web.config.debug  = True
import model
render = web.template.render('templates', base='base')
from config import DETECTANGLE

class OCR:
    """通用OCR识别"""

    def GET(self):
        post = {}
        post['postName'] = 'ocr'##请求地址
        post['height'] = 1000
        post['H'] = 1000
        post['width'] = 600
        post['W'] = 600
        post['uuid'] = uuid.uuid1().__str__()
        return render.ocr(post)

    def POST(self):
        data = web.data()
        data = json.loads(data)
        imgString = data['imgString'].encode().split(b';base64,')[-1]
        imgString = base64.b64decode(imgString)
        jobid = uuid.uuid1().__str__()
        path = '/tmp/{}.jpg'.format(jobid)
        with open(path,'wb') as f:
            f.write(imgString)
        img = Image.open(path).convert("RGB")
        W,H = img.size
        timeTake = time.time()
        _,result,angle= model.model(img,
                                    detectAngle=DETECTANGLE,##是否进行文字方向检测
                                    config=dict(MAX_HORIZONTAL_GAP=100,##字符之间的最大间隔，用于文本行的合并
                                    MIN_V_OVERLAPS=0.7,
                                    MIN_SIZE_SIM=0.7,
                                    TEXT_PROPOSALS_MIN_SCORE=0.1,
                                    TEXT_PROPOSALS_NMS_THRESH=0.3,
                                    TEXT_LINE_NMS_THRESH = 0.99,##文本行之间测iou值
                                    MIN_RATIO=1.0,
                                    LINE_MIN_SCORE=0.2,
                                    TEXT_PROPOSALS_WIDTH=0,
                                    MIN_NUM_PROPOSALS=0,                                               
                ),
                                    leftAdjust=True,##对检测的文本行进行向左延伸
                                    rightAdjust=True,##对检测的文本行进行向右延伸
                                    alph=0.2,##对检测的文本行进行向右、左延伸的倍数
                                    ifadjustDegree=False##是否先小角度调整文字倾斜角度
                                   )
        
        timeTake = time.time()-timeTake
        res = map(lambda x:{'w':x['w'],'h':x['h'],'cx':x['cx'],'cy':x['cy'],'degree':x['degree'],'text':x['text']}, result)
        res = list(res)

        os.remove(path)
        return json.dumps({'res':res,'timeTake':round(timeTake,4)},ensure_ascii=False)




urls = ('/ocr','OCR',)

if __name__ == "__main__":

      app = web.application(urls, globals())
      app.run()
