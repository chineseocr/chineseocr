##环境配置，支持linux\macOs
conda create -n chineseocr python=3.6 pip scipy numpy jupyter ipython ##运用conda 创建python环境
source activate chineseocr
git submodule init && git submodule update
pip install easydict opencv-contrib-python Cython h5py lmdb mahotas pandas requests  -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -U pillow -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install web.py==0.40.dev0
conda install pytorch torchvision -c pytorch
## pip install torch torchvision
pushd detector/utils && sh make.sh && popd


