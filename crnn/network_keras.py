from keras.layers import (Conv2D,BatchNormalization,MaxPool2D,Input,Permute,Reshape,Dense,LeakyReLU,Activation, Bidirectional, LSTM, TimeDistributed)
from keras.models import Model
from keras.layers import ZeroPadding2D
from keras.activations import relu


def keras_crnn(imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False,lstmFlag=True):
    """
    基于pytorch 实现 keras dense ocr
    pytorch lstm 层暂时无法转换为 keras lstm层
    """
    data_format='channels_first'
    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 0]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]
    imgInput = Input(shape=(1,imgH,None),name='imgInput')
    
    def convRelu(i, batchNormalization=False,x=None):
            ##padding: one of `"valid"` or `"same"` (case-insensitive).
            ##nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            if leakyRelu:
                activation = LeakyReLU(alpha=0.2)
            else:
                activation = Activation(relu,name='relu{0}'.format(i))
            
            x = Conv2D(filters=nOut, 
                   kernel_size=ks[i],
                   strides=(ss[i], ss[i]),
                   padding= 'valid' if ps[i]==0 else 'same', 
                   dilation_rate=(1, 1), 
                   activation=None, use_bias=True,data_format=data_format,
                   name='cnn.conv{0}'.format(i)
                   )(x)
            
            if batchNormalization:
                ## torch nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                x = BatchNormalization(epsilon=1e-05,axis=1, momentum=0.1,name='cnn.batchnorm{0}'.format(i))(x)
                
                
            x = activation(x)
            return x
        
    x = imgInput
    x = convRelu(0,batchNormalization=False,x=x)
    
    #x = ZeroPadding2D(padding=(0, 0), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2),name='cnn.pooling{0}'.format(0),padding='valid',data_format=data_format)(x)
    

    x = convRelu(1,batchNormalization=False,x=x)
    #x = ZeroPadding2D(padding=(0, 0), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2),name='cnn.pooling{0}'.format(1),padding='valid',data_format=data_format)(x)
    
    x = convRelu(2, batchNormalization=True,x=x)
    x = convRelu(3, batchNormalization=False,x=x)
    x = ZeroPadding2D(padding=(0, 1), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2),strides=(2,1),padding='valid',name='cnn.pooling{0}'.format(2),data_format=data_format)(x)
    
    x = convRelu(4, batchNormalization=True,x=x)
    x = convRelu(5, batchNormalization=False,x=x)
    x = ZeroPadding2D(padding=(0, 1), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2),strides=(2,1),padding='valid',name='cnn.pooling{0}'.format(3),data_format=data_format)(x)
    x = convRelu(6, batchNormalization=True,x=x)  
    
    x = Permute((3, 2, 1))(x)
    
    x = Reshape((-1,512))(x)

    out = None
    if lstmFlag:
        x = Bidirectional(LSTM(nh, return_sequences=True, use_bias=True,
                               recurrent_activation='sigmoid'))(x)
        x = TimeDistributed(Dense(nh))(x)
        x = Bidirectional(LSTM(nh, return_sequences=True, use_bias=True,
                               recurrent_activation='sigmoid'))(x)
        out = TimeDistributed(Dense(nclass))(x)
    else:
        out = Dense(nclass,name='linear')(x)
    out = Reshape((-1, 1, nclass),name='out')(out)

    return Model(imgInput,out)
