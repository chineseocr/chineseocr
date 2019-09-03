# -*- coding: utf-8 -*-
"""
@author: lywen
"""
import os
import json
import time
import web
import numpy as np
import uuid
from PIL import Image
web.config.debug  = True

filelock='file.lock'
if os.path.exists(filelock):
   os.remove(filelock)

render = web.template.render('templates', base='base')
from config import *
from apphelper.image import union_rbox,adjust_box_to_origin,base64_to_PIL
from application import trainTicket,idcard 
if yoloTextFlag =='keras' or AngleModelFlag=='tf' or ocrFlag=='keras':
    if GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
        import tensorflow as tf
        from keras import backend as K
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.3## GPU最大占用量
        config.gpu_options.allow_growth = True##GPU是否可动态增加
        K.set_session(tf.Session(config=config))
        K.get_session().run(tf.global_variables_initializer())
    
    else:
      ##CPU启动
      os.environ["CUDA_VISIBLE_DEVICES"] = ''

if yoloTextFlag=='opencv':
    scale,maxScale = IMGSIZE
    from text.opencv_dnn_detect import text_detect
elif yoloTextFlag=='darknet':
    scale,maxScale = IMGSIZE
    from text.darknet_detect import text_detect
elif yoloTextFlag=='keras':
    scale,maxScale = IMGSIZE[0],2048
    from text.keras_detect import  text_detect
else:
     print( "err,text engine in keras\opencv\darknet")
     
from text.opencv_dnn_detect import angle_detect

if ocr_redis:
    ##多任务并发识别
    from apphelper.redisbase import redisDataBase
    ocr = redisDataBase().put_values
else:   
    from crnn.keys import alphabetChinese,alphabetEnglish
    if ocrFlag=='keras':
        from crnn.network_keras import CRNN
        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelKerasLstm
            else:
                ocrModel = ocrModelKerasDense
        else:
            ocrModel = ocrModelKerasEng
            alphabet = alphabetEnglish
            LSTMFLAG = True
            
    elif ocrFlag=='torch':
        from crnn.network_torch import CRNN
        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelTorchLstm
            else:
                ocrModel = ocrModelTorchDense
                
        else:
            ocrModel = ocrModelTorchEng
            alphabet = alphabetEnglish
            LSTMFLAG = True
    elif ocrFlag=='opencv':
        from crnn.network_dnn import CRNN
        ocrModel = ocrModelOpencv
        alphabet = alphabetChinese
    else:
        print( "err,ocr engine in keras\opencv\darknet")
     
    nclass = len(alphabet)+1   
    if ocrFlag=='opencv':
        crnn = CRNN(alphabet=alphabet)
    else:
        crnn = CRNN( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=LSTMFLAG,GPU=GPU,alphabet=alphabet)
    if os.path.exists(ocrModel):
        crnn.load_weights(ocrModel)
    else:
        print("download model or tranform model with tools!")
        
    ocr = crnn.predict_job
    
   
from main import TextOcrModel

model =  TextOcrModel(ocr,text_detect,angle_detect)
    

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
        post['billList'] = billList
        return render.ocr(post)

    def POST(self):
        t = time.time()
        data = web.data()
        uidJob = uuid.uuid1().__str__()
        
        data = json.loads(data)
        billModel = data.get('billModel','')
        textAngle = data.get('textAngle',False)##文字检测
        textLine = data.get('textLine',False)##只进行单行识别
        
        imgString = data['imgString'].encode().split(b';base64,')[-1]
        img = base64_to_PIL(imgString)
        if img is not None:
            img = np.array(img)
            
        H,W = img.shape[:2]

        while time.time()-t<=TIMEOUT:
            if os.path.exists(filelock):
                continue
            else:
                with open(filelock,'w') as f:
                    f.write(uidJob)
                                                
                if textLine:
                    ##单行识别
                    partImg = Image.fromarray(img)
                    text    = crnn.predict(partImg.convert('L'))
                    res =[ {'text':text,'name':'0','box':[0,0,W,0,W,H,0,H]} ]
                    os.remove(filelock)
                    break
                        
                else:
                    detectAngle = textAngle
                    result,angle= model.model(img,
                                                scale=scale,
                                                maxScale=maxScale,
                                                detectAngle=detectAngle,##是否进行文字方向检测，通过web传参控制
                                                MAX_HORIZONTAL_GAP=100,##字符之间的最大间隔，用于文本行的合并
                                                MIN_V_OVERLAPS=0.6,
                                                MIN_SIZE_SIM=0.6,
                                                TEXT_PROPOSALS_MIN_SCORE=0.1,
                                                TEXT_PROPOSALS_NMS_THRESH=0.3,
                                                TEXT_LINE_NMS_THRESH = 0.99,##文本行之间测iou值
                                                LINE_MIN_SCORE=0.1,
                                                leftAdjustAlph=0.01,##对检测的文本行进行向左延伸
                                                rightAdjustAlph=0.01,##对检测的文本行进行向右延伸
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
                        
                    os.remove(filelock)
                    break
            
        
        timeTake = time.time()-t
         
        return json.dumps({'res':res,'timeTake':round(timeTake,4)},ensure_ascii=False)
        

urls = ('/ocr','OCR',)

if __name__ == "__main__":

      app = web.application(urls, globals())
      app.run()
