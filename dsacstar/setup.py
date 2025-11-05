"""Build script for the dsacstar C++ extension."""

import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Allow overriding OpenCV locations via environment variables.
opencv_inc_dir = os.environ.get("OPENCV_INCLUDE_DIR", "")  # directory containing OpenCV header files
opencv_lib_dir = os.environ.get("OPENCV_LIBRARY_DIR", "")  # directory containing OpenCV library files

# If not explicitly provided, try to locate OpenCV in the current Conda environment.
conda_env = os.environ.get("CONDA_PREFIX", "")

if len(conda_env) > 0 and len(opencv_inc_dir) == 0 and len(opencv_lib_dir) == 0:
    print("Detected active conda environment:", conda_env)

    opencv_inc_dir = conda_env + "/include/opencv4"
    opencv_lib_dir = conda_env + "/lib/"

    print("Assuming OpenCV dependencies in:")
    print(opencv_inc_dir)
    print(opencv_lib_dir)

if len(opencv_inc_dir) == 0:
    print("Error: You have to provide an OpenCV include directory.")
    print("Set $OPENCV_INCLUDE_DIR or ensure a conda env is active.")
    sys.exit(1)
if len(opencv_lib_dir) == 0:
    print("Error: You have to provide an OpenCV library directory.")
    print("Set $OPENCV_LIBRARY_DIR or ensure a conda env is active.")
    sys.exit(1)

setup(
    name="dsacstar",
    packages=["dsacstar"],
    ext_modules=[
        CppExtension(
            name="dsacstar._C",
            sources=["dsacstar.cpp", "thread_rand.cpp"],
            include_dirs=[opencv_inc_dir],
            library_dirs=[opencv_lib_dir],
            libraries=["opencv_core", "opencv_calib3d"],
            extra_compile_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
