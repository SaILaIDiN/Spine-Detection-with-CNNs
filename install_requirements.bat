@echo off
setlocal

pip install --no-cache-dir --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install openmim

REM On Python 3.9 no pre-built package exists and it will be compiled, taking ~10 minutes
mim install mmcv-full==1.6
pip install .

endlocal
