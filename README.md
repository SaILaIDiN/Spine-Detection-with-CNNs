# Spine-Detection-with-CNNs
Code for detecting dendritic spines in three dimensions, retraining and evaluating models.

Structure of this guide:
1. Installation
2. Folder structure
3. Prediction on 2D-images
4. Prediction and tracking on 3D-images
   - File format for prediction and tracking csv
5. Re-Training with new dataset
   - Prepare dataset
   - Training
   - Prepare model for inference
6. Model evaluation

## Installation
Some packages are listed in the `requirements.txt` file. To install all necessary packages correctly using pip, simply run
```
./install_requirements.sh
```
The model, training and evaluation images are not saved in GitHub, but are available for download. The data can automatically be downloaded and extracted into the correct folders using the following command:
```
sh download_data.sh
```

If more control over the data is required, the model can be downloaded [here](https://drive.google.com/uc?export=download&id=1IqXEYAbruormi9g354a1MtugQJQiZKGL) and the images with their labels can be downloaded [here](https://drive.google.com/uc?export=download&id=1yi2tQ-0oJhElaSUDFn_UpZ-bUO0bH3_N). The model and images should then be extracted into the `own_models/default_model` and `data/raw` folder.

## Folder structure
This github repository provides all necessary files to predict and track dendritic spines as described in [this paper](https://www.biorxiv.org/content/10.1101/2023.01.08.522220.full.pdf). Retraining on another dataset is possible as well. The mainly relevant files and structures of this repository are:
```
|-- src/spine_detection
|   |-- configs
|   |-- train_mmdet.py
|   |-- predict_mmdet.py
|   |-- tracking_mmdet.py
|   |-- evaluate_tracking_mmdet.py
|-- data
|   |-- default_annotations
|   `-- raw
|-- references/mmdetection
|   |-- checkpoints
|   `-- configs
|-- output
|   |-- prediction
|   |   |-- custom_model
|   |   |   |-- csvs_mmdet
|   |   |   `-- images_mmdet
|   |   `-- default_model
|   |       |-- csvs_mmdet
|   |       `-- images_mmdet
|   `-- tracking
|       |-- custom_model
|       |   |-- exp1
|       |   |   `-- data_tracking.csv
|       |   |-- exp2
|       |   |   `-- data_tracking.csv
|       `-- default_model
|           `-- data_tracking.csv
|-- tutorial_exps
|   |-- custom_model
|   |   |-- exp1
|   |   |-- exp2
|   |   |   |-- epoch_1.pth
|   |   |   |-- ...
|   |   |   |-- epoch_n.pth
|   |   |   |-- latest.pth
|   |   |   `-- log.json
|   `-- default_model
|       `-- model.pth
|-- install_requirements.sh
|-- download_data.sh
|-- requirements.txt
`-- setup.py
```
## Inference
```bash
python src/spine_detection/predict_mmdet.py \
    -i "data/raw/person1/SR052N1D1day1*.png" \
    -m Cascade_RCNN \
    -pc lr_0.0005_warmup_None_momentum_None_L2_None
```

## Tracking
```bash
python src/spine_detection/tracking_mmdet.py \
    -i "data/raw/person1/SR052N1D1day1*.png"\
    -m Cascade_RCNN \
    -pc lr_0.0005_warmup_None_momentum_None_L2_None
```
## Evaluate tracking
```bash
python src/spine_detection/evaluate_tracking_mmdet.py \
    -m Cascade_RCNN \
    -pc lr_0.0005_warmup_None_momentum_None_L2_None \
    -tr data_tracking_default_aug_False_epoch_1_theta_0.5_delta_0.5_Test.csv
```
## Training
```bash
python src/spine_detection/train_mmdet.py
```
