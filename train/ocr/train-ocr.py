import os
import numpy as np
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from warpctc_pytorch import CTCLoss
from train.ocr.dataset import PathDataset,randomSequentialSampler,alignCollate
from glob import glob
from sklearn.model_selection import train_test_split

roots = glob('./train/data/ocr/*/*.jpg')

alphabetChinese = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

trainP,testP = train_test_split(roots,test_size=0.1)##此处未考虑字符平衡划分
traindataset = PathDataset(trainP,alphabetChinese)
testdataset = PathDataset(testP,alphabetChinese)

batchSize = 32
workers = 1
imgH = 32
imgW = 280
keep_ratio = True
cuda = True
ngpu = 1
nh =256
sampler = randomSequentialSampler(traindataset, batchSize)
train_loader = torch.utils.data.DataLoader(
    traindataset, batch_size=batchSize,
    shuffle=False, sampler=None,
    num_workers=int(workers),
    collate_fn=alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

train_iter = iter(train_loader)