#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 02:26:46 2019
run ocr job
@author: chineseocr
"""
import os
import subprocess
jobNum = 4##任务并行数
for i in range(jobNum):
         pid = os.system('nohup python ocrjob.py >/tmp/ocr-{}.log 2>&1 &'.format(i))
         print(i,pid)
