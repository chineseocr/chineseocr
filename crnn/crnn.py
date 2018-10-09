#coding:utf-8
import sys
sys.path.insert(1, "./crnn")
import torch
import torch.utils.data
from torch.autograd import Variable 
from crnn import util
from crnn import dataset
from crnn.models import crnn as crnn
from crnn import keys
from collections import OrderedDict
from config import ocrModel,LSTMFLAG,GPU
from config import chinsesModel
def crnnSource():
    if chinsesModel:
        alphabet = keys.alphabetChinese
    else:
        alphabet = keys.alphabetEnglish
        
    converter = util.strLabelConverter(alphabet)
    if torch.cuda.is_available() and GPU:
        model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1,lstmFlag=LSTMFLAG).cuda()##LSTMFLAG=True crnn 否则 dense ocr
    else:
        model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1,lstmFlag=LSTMFLAG).cpu()
    
    state_dict = torch.load(ocrModel,map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.','') # remove `module.`
        new_state_dict[name] = v
    # load params
   
    model.load_state_dict(new_state_dict)
    model.eval()
    
    return model,converter

##加载模型
model,converter = crnnSource()

def crnnOcr(image):
       """
       crnn模型，ocr识别
       @@model,
       @@converter,
       @@im
       @@text_recs:text box

       """
       scale = image.size[1]*1.0 / 32
       w = image.size[0] / scale
       w = int(w)
       #print "im size:{},{}".format(image.size,w)
       transformer = dataset.resizeNormalize((w, 32))
       if torch.cuda.is_available() and GPU:
           image = transformer(image).cuda()
       else:
           image = transformer(image).cpu()
            
       image = image.view(1, *image.size())
       image = Variable(image)
       model.eval()
       preds = model(image)
       _, preds = preds.max(2)
       preds = preds.transpose(1, 0).contiguous().view(-1)
       preds_size = Variable(torch.IntTensor([preds.size(0)]))
       sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

       return sim_pred
       

