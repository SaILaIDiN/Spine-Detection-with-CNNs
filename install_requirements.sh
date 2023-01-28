
pip install --no-cache-dir --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install openmim && mim install mmcv-full==1.6
pip install .