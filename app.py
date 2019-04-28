# -*- coding: utf-8 -*-
"""
@author: lywen
"""
import os
import cv2
import json
import time
import uuid
import base64
import web
from PIL import Image
web.config.debug  = True
import model
render = web.template.render('templates', base='base')
from config import DETECTANGLE
from apphelper.image import union_rbox,adjust_box_to_origin
from application import trainTicket,idcard 


billList = ['通用OCR','火车票','身份证']

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
        post['billList'] = billList
        return render.ocr(post)

    def POST(self):
        data = web.data()
        data = json.loads(data)
        billModel = data.get('billModel','')
        textAngle = data.get('textAngle',False)##文字检测
        textLine = data.get('textLine',False)##只进行单行识别
        
        imgString = data['imgString'].encode().split(b';base64,')[-1]
        imgString = base64.b64decode(imgString)
        jobid = uuid.uuid1().__str__()
        path = 'test/{}.jpg'.format(jobid)
        with open(path,'wb') as f:
            f.write(imgString)
        img = cv2.imread(path)##GBR
        H,W = img.shape[:2]
        timeTake = time.time()
        if textLine:
            ##单行识别
            partImg = Image.fromarray(img)
            text = model.crnnOcr(partImg.convert('L'))
            res =[ {'text':text,'name':'0','box':[0,0,W,0,W,H,0,H]} ]
        else:
            detectAngle = textAngle
            _,result,angle= model.model(img,
                                        detectAngle=detectAngle,##是否进行文字方向检测，通过web传参控制
                                        config=dict(MAX_HORIZONTAL_GAP=50,##字符之间的最大间隔，用于文本行的合并
                                        MIN_V_OVERLAPS=0.6,
                                        MIN_SIZE_SIM=0.6,
                                        TEXT_PROPOSALS_MIN_SCORE=0.1,
                                        TEXT_PROPOSALS_NMS_THRESH=0.3,
                                        TEXT_LINE_NMS_THRESH = 0.7,##文本行之间测iou值
                                                ),
                                        leftAdjust=True,##对检测的文本行进行向左延伸
                                        rightAdjust=True,##对检测的文本行进行向右延伸
                                        alph=0.01,##对检测的文本行进行向右、左延伸的倍数
                                       )



            if billModel=='' or billModel=='通用OCR' :
                result = union_rbox(result,0.2)
                res = [{'text':x['text'],
                        'name':str(i),
                        'box':{'cx':x['cx'],
                               'cy':x['cy'],
                               'w':x['w'],
                               'h':x['h'],
                               'angle':x['degree']

                              }
                       } for i,x in enumerate(result)]
                res = adjust_box_to_origin(img,angle, res)##修正box

            elif billModel=='火车票':
                res = trainTicket.trainTicket(result)
                res = res.res
                res =[ {'text':res[key],'name':key,'box':{}} for key in res]

            elif billModel=='身份证':

                res = idcard.idcard(result)
                res = res.res
                res =[ {'text':res[key],'name':key,'box':{}} for key in res]
            
        
        timeTake = time.time()-timeTake
         
        
        os.remove(path)
        return json.dumps({'res':res,'timeTake':round(timeTake,4)},ensure_ascii=False)
        

urls = ('/ocr','OCR',)

if __name__ == "__main__":

      app = web.application(urls, globals())
      app.run()
