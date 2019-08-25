## 环境配置，支持linux\macOs      
conda create -n chineseocr python=3.6 pip scipy numpy jupyter ipython ##运用conda 创建python环境      
source activate chineseocr      
git submodule init && git submodule update      
pip install easydict opencv-contrib-python==4.0.0.21 Cython h5py lmdb mahotas pandas requests bs4 matplotlib lxml -i https://pypi.tuna.tsinghua.edu.cn/simple/        
pip install -U pillow -i https://pypi.tuna.tsinghua.edu.cn/simple/      
pip install keras==2.1.5 tensorflow==1.8 tensorflow-gpu==1.8      
pip install web.py==0.40.dev0      redis
conda install pytorch torchvision -c pytorch          
## pip install torch torchvision   
