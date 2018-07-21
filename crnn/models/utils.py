#!/usr/bin/python
# encoding: utf-8

import torch.nn as nn
import torch.nn.parallel


def data_parallel(model, input, ngpu):
    if isinstance(input.data, torch.cuda.FloatTensor) and ngpu > 1:
        output = nn.parallel.data_parallel(model, input, range(ngpu))
    else:
        output = model(input)
    return output
