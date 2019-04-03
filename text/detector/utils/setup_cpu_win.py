from Cython.Build import cythonize
import os
from os.path import join as pjoin
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = cythonize([
    Extension(
        "utils.cython_nms",
        sources=["cython_nms.pyx"],
        language="c",
        include_dirs = [numpy_include],
        library_dirs=[],
        libraries=[],
        extra_compile_args=[],
        extra_link_args=[]

         # extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
    ),
])

setup(
    ext_modules = ext_modules
    # ,
    # cmdclass = {'build_ext': build_ext},
)

