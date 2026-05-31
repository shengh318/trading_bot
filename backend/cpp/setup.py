from setuptools import setup, Extension
import pybind11
import sys

ext = Extension(
    "pairs_engine",
    sources=["pairs_engine.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17"],
)

setup(
    name="pairs_engine",
    version="0.1.0",
    ext_modules=[ext],
    zip_safe=False,
)
