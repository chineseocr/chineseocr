#coding:utf-8
import torch
import torch.utils.data
from torch.autograd import Variable 
from crnn import util
from crnn import dataset
from crnn.network import CRNN
from crnn import keys
from collections import OrderedDict
from config import ocrModel,LSTMFLAG,GPU
from config import chinsesModel
def crnnSource():
    if chinsesModel:
        alphabet = keys.alphabetChinese##中英文模型
    else:
        alphabet = keys.alphabetEnglish##英文模型
        
    converter = util.strLabelConverter(alphabet)
    if torch.cuda.is_available() and GPU:
        model = CRNN(32, 1, len(alphabet)+1, 256, 1,lstmFlag=LSTMFLAG).cuda()##LSTMFLAG=True crnn 否则 dense ocr
    else:
        model = CRNN(32, 1, len(alphabet)+1, 256, 1,lstmFlag=LSTMFLAG).cpu()
    
    trainWeights = torch.load(ocrModel,map_location=lambda storage, loc: storage)
    modelWeights = OrderedDict()
    for k, v in trainWeights.items():
        name = k.replace('module.','') # remove `module.`
        modelWeights[name] = v
    # load params
   
    model.load_state_dict(modelWeights)
    model.eval()

    return model,converter

##加载模型
model,converter = crnnSource()

def crnnOcr(image):
       """
       crnn模型，ocr识别
       image:PIL.Image.convert("L")
       """
       scale = image.size[1]*1.0 / 32
       w = image.size[0] / scale
       w = int(w)
       transformer = dataset.resizeNormalize((w, 32))
       if torch.cuda.is_available() and GPU:
           image   = transformer(image).cuda()
       else:
           image   = transformer(image).cpu()
            
       image       = image.view(1, *image.size())
       image       = Variable(image)
       model.eval()
       preds       = model(image)
       _, preds    = preds.max(2)
       preds       = preds.transpose(1, 0).contiguous().view(-1)
       preds_size  = Variable(torch.IntTensor([preds.size(0)]))
       sim_pred    = converter.decode(preds.data, preds_size.data, raw=False)

       return sim_pred
       

