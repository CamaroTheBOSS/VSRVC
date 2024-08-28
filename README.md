## VSRVC
Double task model for video compression and super-resolution

### Prerequisites
- python 3.8
- pytorch 1.8.1
- LibMTL
- CompressAI

### Environment
1. Install torch
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

2. Install external packages: LibMTL and CompressAI

LibMTL
```
# LibMTL
mkdir external
cd external
git clone git@github.com:median-research-group/LibMTL.git
cd LibMTL
pip install -r requirements.txt
pip install -e .
```

CompressAI (Windows)
```
# CompressAI
cd ..
git clone git@github.com:InterDigitalInc/CompressAI.git
cd CompressAI
pip install wheel
python setup.py bdist_wheel --dist-dir dist/
pip install dist/compressai-*.whl
```

CompressAI (Linux)
```
pip install compressai
```

3. Install VSRVC remaining dependencies 
```
pip install wandb kornia==0.6.8 torchmetrics==1.2.1 opencv-python
```
Environment was tested on both Linux and Windows with CUDA 12.1 (tested on 22.08.2024)
### Training
```commandline
python main.py --model_type vsrvc_shallow --vimeo_path ../Datasets/VIMEO90k --weighting EW --save_path "./weights" --lmbda 128 --epochs 30 --scale 4 --mode train --num_workers 1 --optim adamw --seed 777 --scheduler mycos --vsr --vc
```
Dataset should have structure:
```
vimeo_root
    train
        061
        ...
        070
    sep_testlist.txt
    sep_trainlist.txt
```