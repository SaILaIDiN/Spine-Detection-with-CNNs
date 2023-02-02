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
If you have downloaded our custom model and just want to do inference, you just need to use the following command:
```bash
python src/spine_detection/predict_mmdet.py \
    --input "data/raw/person1/SR052N1D1day1*.png" \
    --model Cascade_RCNN \
    --param_config lr_0.0005_warmup_None_momentum_None_L2_None \
    --model_epoch latest \
    --save_images
```
You should replace the `input` flag with the path to the files you want to do inference on. Your path needs to be either a file or a path to multiple files using wildcards `*`.

The model that will be used for inference must be found at the path `tutorial_exps/<model>/<param_config>/<model_epoch>`.

Activating the flag `save_images` makes sure that not only the csv output but also the images are saved.

The output can then be found by default under `output/prediction/<model>/<param_config>/` with two folders `csvs_mmdet` and `images_mmdet`. However, you can also define your own output path using the `--output` flag.

If you want to change the confidence threshold, adjust the flag `--delta` (default is 0.5).

## Tracking
If you are not only interested in detecting spines on single 2D images but in a 3D stack consisting of multiple images, you can use the tracking command as follows:
```bash
python src/spine_detection/tracking_mmdet.py \
    --input "data/raw/person1/SR052N1D1day1*.png"\
    --model Cascade_RCNN \
    --param_config lr_0.0005_warmup_None_momentum_None_L2_None \
    --model_epoch latest \
    --save_images
```
The basic parameters are the same as with the `predict_mmdet.py` script. However, you have more options to customize the tracker, e.g. which metric should be used for box overlap between different images.

Similarly, the output can be found by default under `output/tracking/<model>/<param_config>/` with a folder `images` and a file `data_tracking_<model-type>_aug_<use_aug>_<model_epoch>_theta_<theta>_delta_<delta>_<input_mode>.csv`. All these parameters can be explored by looking into the code or run the tracking file with the `--help` flag.

### Evaluate tracking
After having tracked the dendritic spines over 3D stacks, you can evaluate the models performance when the groundtruth is also available for the analyzed data as follows:
```bash
python src/spine_detection/evaluate_tracking_mmdet.py \
    --model Cascade_RCNN \
    --param_config lr_0.0005_warmup_None_momentum_None_L2_None \
    --tracking data_tracking_default_aug_False_epoch_1_theta_0.5_delta_0.5_Test.csv \
    --gt_file output/tracking/GT/data_tracking_max_wo_offset.csv
```
For the `tracking` flag you should choose the name of the above generated `data_tracking_*.csv` file. The `gt_file` argument should correspond to a csv file consisting of tracked labeled data. (Easily creating ground truth tracked file is not available yet)

## Training
For every kind of training you need to prepare your data correctly. Your data should be in the same format as the data you could download from us, especially the labels and the folder structure. If you want to create and use your own dataset, save the train, val and test labels somewhere inside the `data` folder together with your images. It is not important if these are in the same folder as the labels but the path in the labels should be correct and start with `data/`.

After having downloaded our custom models you can continue training them with your own data with the following command:
```bash
python src/spine_detection/train_mmdet.py \
    --model_type default \
    --checkpoint lr_0.0005_warmup_None_momentum_None_L2_None/latest.pth \
    --resume
```
In that case the script is searching for a model file (`.pth`) inside `tutorial_exps/<model_name>/<checkpoint>`. `model_name` is the value of `model_paths.<model_type>` in the config file `configs/model_config_paths.yaml`. If you have used our `download_data.sh` script, the models should already have the correct paths.

### Train on pretrained model from mmdetection modelzoo
If you want to train on any pretrained model you first need to choose which pretrained model you want to use and download it into `references/mmdetection/checkpoint`. You now have downloaded a model with name `<model_type>_<param_config>_<date>-<identifier>.pth`. If you can find any `model_paths` with `base_config` equal to `<model>_<param_config>_coco` in `configs/model_config_paths.yaml` it's fine but if not you need to create a new config file `cfg_file=references/mmdetection/configs/<model>/<model>_<param_config>_coco_SPINE.py`. The config file should look like the the corresponding python file without the `SPINE` addition.

Now take a look into all the `_base_` configs the original config file is referring to. At some point it will refer to a file inside `references/mmdetection/configs/_base_/datasets`. You can copy this into your newly created file `cfg_file` and replace the following fields:

```python
data_root = 'data/raw'
data.train.ann_file = 'data/default_annotations/data_train.csv'
data.val.ann_file = 'data/default_annotations/data_val.csv'
data.test.ann_file = 'data/default_annotations/data_test.csv'
work_dir = 'tutorial_exps'
```

Afterwards you need to add the following entry inside the `configs/model_config_paths.yaml` under `model_paths`:
```yaml
model_paths:
    <custom_model_type>:
        base_checkpoint: <model>
        base_config: <model>/<model>_<param_config>_coco
```
where `<custom_model_type>` can be any custom name you would like to use. The training can then be started with

```bash
python src/spine_detection/train_mmdet.py \
    --model_type <custom_model_type>
```