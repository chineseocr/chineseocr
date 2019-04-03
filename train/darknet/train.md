## 下载ICDR2019数据集 地址 http://rrc.cvc.uab.es/?ch=14
##  百度云地址:https://pan.baidu.com/s/1fmOpTYFmZ4f2UxGDsmKVJA  密码:sh4e

## 下载 darknet预训练模型   
wget https://pjreddie.com/media/files/darknet53.conv.74    
## 修改  train/darknet/data-ready.py 
line166 ~line 168   
dataRoot    = '/tmp/ICDR2019/'##icdr2019 数据所在目录     
darknetRoot = '/tmp/darknet'##darknet目录     
wP          = '/tmp/darknet53.conv.74'##darknet 预训练模型权重     

执行 python train/darknet/data-ready.py  生成darknet格式训练数据   
训练模型 : sh train/darknet/train.sh   
