#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 02:02:29 2019
job
@author: chineseocr
"""
from apphelper.redisbase import redisDataBase
from config import *
from crnn.keys import alphabetChinese,alphabetEnglish

if ocrFlag=='keras':
    if GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
        import tensorflow as tf
        from keras import backend as K
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.1## GPU最大占用量
        config.gpu_options.allow_growth = True##GPU是否可动态增加
        K.set_session(tf.Session(config=config))
        K.get_session().run(tf.global_variables_initializer())

    else:
        ##CPU启动
        os.environ["CUDA_VISIBLE_DEVICES"] = ''

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

else:
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

nclass = len(alphabet)+1

ocr = CRNN( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=LSTMFLAG,GPU=GPU,alphabet=alphabet)
if os.path.exists(ocrModel):
    ocr.load_weights(ocrModel)
else:
    print("download model or tranform model with tools!")

if __name__=='__main__':
    redisJob = redisDataBase()
    while True:
        redisJob.get_job(ocr.predict)

