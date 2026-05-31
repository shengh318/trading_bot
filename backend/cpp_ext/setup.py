from setuptools import setup, Extension
import pybind11
import numpy as np

ext = Extension(
    "cpp_accel",
    ["module.cpp"],
    include_dirs=[
        pybind11.get_include(),
        np.get_include(),
    ],
    language="c++",
    extra_compile_args=["/O2", "/std:c++17", "/EHsc"],
)

setup(
    name="cpp_accel",
    version="1.0.0",
    description="C++ acceleration module for TraderBot",
    ext_modules=[ext],
)
