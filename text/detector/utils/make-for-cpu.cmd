@echo off
cython cython_nms.pyx
python setup_cpu_win.py build_ext --inplace
MOVE /Y utils\*.* .\
RMDIR /S /Q build
RMDIR /S /Q utils
