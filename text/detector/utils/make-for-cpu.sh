cython cython_nms.pyx
python setup_cpu.py build_ext --inplace
mv utils/* ./
rm -rf build
rm -rf utils

