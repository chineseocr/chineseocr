from . import cython_nms
try:
  from . import gpu_nms
except:
   gpu_nms = cython_nms
   
