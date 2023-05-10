@echo off
setlocal

python -m pip install --upgrade pip
pip install --no-cache-dir --upgrade wheel setuptools
pip install torch torchvision torchaudio
pip install -r requirements.txt

@REM On Python 3.9 no pre-built package exists and it will be compiled, taking ~10 minutes
pip install mmcv-full==1.6 @REM -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
pip install .

endlocal
