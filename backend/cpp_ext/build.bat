@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0..\.."
set PYTHONPATH=%~dp0
.venv\Scripts\python -c "import sys; sys.path.insert(0, '%~dp0'); from setuptools import setup; exec(open('%~dp0setup.py').read())" build_ext --inplace --build-lib "%~dp0"
