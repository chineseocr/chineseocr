import random
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms


class PathDataset(Dataset):

    def __init__(self, jpgPaths,alphabetChinese, transform=None, target_transform=None):
        """
        加载本地目录图片
        """
        
        self.jpgPaths = jpgPaths
        self.nSamples = len(self.jpgPaths)
        self.alphabetChinese = alphabetChinese
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index >= len(self):
            index=0
            
        imP = self.jpgPaths[index]
        txtP = imP.replace('.jpg','.txt')
        im = Image.open(imP).convert('L')
        with open(txtP)  as f:
            label = f.read().strip()
            
        label = ''.join([ x for x in label if x in self.alphabetChinese ])
            
        if self.transform is not None:
         
                im = self.transform(im)
            
        if self.target_transform is not None:
                    label = self.target_transform(label)
        
                
        return (im, label)    
        
    
class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        size = self.size
        imgW,imgH = size
        scale = img.size[1]*1.0 / imgH
        w     = img.size[0] / scale
        w     = int(w)
        img   = img.resize((w,imgH),self.interpolation)
        w,h   = img.size
        if w<=imgW:
            newImage       = np.zeros((imgH,imgW),dtype='uint8')
            newImage[:]    = 255
            newImage[:,:w] = np.array(img)
            img            = Image.fromarray(newImage)
        else:
            img   = img.resize((imgW,imgH),self.interpolation)   
        #img = (np.array(img)/255.0-0.5)/0.5
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size )
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail )
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
