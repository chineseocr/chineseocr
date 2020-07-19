FROM ubuntu
MAINTAINER https://github.com/chineseocr/chineseocr
LABEL version="1.0"
EXPOSE 8080
RUN apt-get update && apt-get install -y libsm6 libxrender1 libxext-dev gcc
##下载Anaconda3 python 环境安装包 放置在chineseocr目录 url地址https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
WORKDIR /chineseocr
ADD https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
RUN sh -c 'yes | sh Anaconda3-2019.03-Linux-x86_64.sh' && rm Anaconda3-2019.03-Linux-x86_64.sh
RUN /root/anaconda3/bin/conda install -y python=3.6
RUN /root/anaconda3/bin/pip install easydict opencv-contrib-python==3.4.2.16 Cython h5py pandas requests bs4 matplotlib lxml -U pillow keras==2.1.5 tensorflow==1.8 web.py==0.40.dev0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ && /root/anaconda3/bin/pip cache purge
RUN /root/anaconda3/bin/conda install -y pytorch-cpu torchvision-cpu -c pytorch
COPY . .
#RUN cd /chineseocr/text/detector/utils && sh make-for-cpu.sh
#RUN conda clean -p
#RUN conda clean -t
