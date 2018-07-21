# -*- coding: utf-8 -*-
"""
@author: lywen
todo:通用函数文件
"""
import datetime as dt

def get_date():
    """
    获取当前时间
    """
    try:
        now = dt.datetime.now()
        nowString = now.strftime('%Y%m%d')
    except:
        nowString = '000000'
    return nowString

def strdate_to_date(string,format='%Y-%m-%d %H:%M:%S'):
    try:
        return dt.datetime.strptime(string,format)
    except:
        return dt.datetime.now()
    
def diff_time(beginDate,endDate,format='%Y-%m-%d %H:%M:%S'):
    str1Date = strdate_to_date(beginDate,format)
    str2Date = strdate_to_date(endDate,format)
    times    = str2Date - str1Date
    return   times.total_seconds()
    
def get_now():
    """
    获取当前时间
    """
    try:
        now = dt.datetime.now()
        nowString = now.strftime('%Y-%m-%d %H:%M:%S')
    except:
        nowString = '00-00-00 00:00:00'
    return nowString

