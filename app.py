# -*- coding: utf-8 -*-
"""
@author: lywen
"""
import os
from PIL import Image
import json
import time
import uuid
import numpy as np
import sys
import base64

import web

web.config.debug  = True
from apphelper.image import convert_image,read_url_img,string_to_array,array_to_string,base64_to_array
import model
render = web.template.render('templates', base='base')


class OCR:
    """通用OCR识别"""

    def GET(self):
        post = {}
        post['postName'] = u'ocr'##请求地址
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
        _,result,angle= model.model(img,detectAngle=True,config=dict(MAX_HORIZONTAL_GAP=200,
                MIN_V_OVERLAPS=0.6,
                MIN_SIZE_SIM=0.6,
                TEXT_PROPOSALS_MIN_SCORE=0.2,
                TEXT_PROPOSALS_NMS_THRESH=0.3,
                TEXT_LINE_NMS_THRESH = 0.99,
                MIN_RATIO=1.0,
                LINE_MIN_SCORE=0.2,
                TEXT_PROPOSALS_WIDTH=5,
                MIN_NUM_PROPOSALS=0,
                textmodel = 'opencv_dnn_detect'                                                     
                ),
                leftAdjust=True,rightAdjust=True,alph=0.1)

        res = map(lambda x:{'w':x['w'],'h':x['h'],'cx':x['cx'],'cy':x['cy'],'degree':x['degree'],'text':x['text']}, result)
        res = list(res)

        os.remove(path)
        return json.dumps(res,ensure_ascii=False)




urls = (u'/ocr',u'OCR',
       )

if __name__ == "__main__":

      app = web.application(urls, globals())
      app.run()
