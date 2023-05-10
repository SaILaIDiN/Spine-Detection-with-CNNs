# Spine-Detection-with-CNNs

Code for detecting dendritic spines in three dimensions, retraining and evaluating models.

Structure of this guide:

1. [Installation](#installation)
2. [Folder structure](#folder-structure)
3. [Prediction on 2D-images](#predictioninference-on-2d)
4. [Prediction and tracking on 3D-images](#trackingprediction-on-3d)
5. [Evaluate model on 3D-images](#evaluate-tracking)
6. [Re-Training with new dataset](#training)
7. [FaQ](#faq)

## Installation

Python 3.8 is recommended. Using a CUDA enabled GPU is recommended to get fast prediction times and converge fast during training. However all the code works with only CPU as well you just need to choose the `cpu` versions for the installation and add `--device cpu` on all python commands described in the sections below.
Some packages are listed in the `requirements.txt` file. To install all necessary packages correctly using pip, simply run

```bash
./install_requirements.sh
OR
./install_requirements_cpu.sh
```

in Linux bash or

```bash
install_requirements.bat
OR
install_requirements_cpu.bat
```

in Windows command line. Note that this may take a few minutes, especially collecting, compiling and installing mmcv-full may take more than ten minutes, so please be patient.

The current best models mentioned in [this paper](https://www.biorxiv.org/content/10.1101/2023.01.08.522220.full.pdf) as well as training and evaluation images and labels are not saved in GitHub directly, but are available for download. All the data can automatically be downloaded and extracted into the correct folders using the Linux shell script

```bash
sh download_data.sh
```

or alternatively using the python script on any operating system

```bash
python src/spine_detection/utils/prepare_data.py
```

If you like to have more control over the data, you can download

- the images and labels [here](https://drive.google.com/uc?id=1yi2tQ-0oJhElaSUDFn_UpZ-bUO0bH3_N)
- the Cascade RCNN model [here](https://drive.google.com/uc?id=1eLeqafL4UPM3-uPSV37SvEo00ONM-1tD)
- and the Faster RCNN model [here](https://drive.google.com/uc?id=1OnbBMdaOsc9-TPFkOTr4geCNLT5Dv7w3).

Generally all python commands described in the sections below have the optional parameter `--device` which determines the device on which the prediction, training or model evaluation happens (`cpu` or `cuda:<gpu-id>`) and the optional parameter `--log_level` which determines the log level of the output (`debug`, `info`, `warning` or `error`).

By default `--device cuda:0` and `--log_level info` are used.

## Folder structure

This github repository provides all necessary files to predict and track dendritic spines as described in [this paper](https://www.biorxiv.org/content/10.1101/2023.01.08.522220.full.pdf). Retraining on another dataset is possible as well. The mainly relevant files of this repository are structured as follows:

```
|-- src/spine_detection
|   |-- utils
|   |   `-- prepare_data.py
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
|   |   |-- ...
|   |   `-- expk
|   |       |-- epoch_1.pth
|   |       |-- ...
|   |       |-- epoch_n.pth
|   |       |-- latest.pth
|   |       `-- log.json
|   |-- ...
|   `-- Cascade_RCNN_model
|       |-- lr_0.001_warmup_None_momentum_0.6_L2_3e-06_run_1
|       |-- ...
|       `-- lr_0.001_warmup_None_momentum_0.6_L2_3e-06_run_4
|           `-- epoch_10.pth
|-- install_requirements.sh
|-- download_data.sh
|-- requirements.txt
`-- setup.py
```

## Prediction/Inference on 2D

If you want to simply detect spines on images using our model and you have already downloaded our custom model (see section [Installation](#installation)), you just need to use the following command in your command line:

```bash
python src/spine_detection/predict_mmdet.py \
    --input "data/raw/person1/SR052N1D1day1*.png" \
    --model Cascade_RCNN_model \
    --model_type Cascade-RCNN \
    --param_config lr_0.001_warmup_None_momentum_0.6_L2_3e-06_run_1 \
    --model_epoch epoch_10 \
    --save_images
```

You should replace the `input` flag with the path to the files you want to do detect spines on. Your path needs to be either a file or a path to multiple files using wildcards `*`.

The model that will be used for prediction must be found at the path `tutorial_exps/<model>/<param_config>/<model_epoch>` inside this model repository. After following the Installation instructions, there should already be models available under the two folders `Cascade_RCNN_model` and `Faster_RCNN_data_augmentation` inside the `tutorial_exps` folder.

The config that is used for loading the model can be found in the `configs/model_config_paths.yaml` file under `model_paths.<model_type>.base_config`. This needs to be adjusted accordingly if you use different configs than provided in the mentioned config file.

**Note: Be careful with the `--model` and `--model_type` flag. For using the default Faster RCNN model, you need to set `--model Faster_RCNN_data_augmentation` but `--model_type Faster-RCNN` according to the config file `configs/model_configs_paths.yaml`.**

Activating the flag `save_images` makes sure that not only the csv output but also the images are saved.

The output can then be found by default under `output/prediction/<model>/<param_config>/` with two folders `csvs_mmdet` and `images_mmdet`. However, you can also define your own output path using the `--output` flag.

If you want to change the confidence threshold, adjust the flag `--delta` (default is 0.5).

For more information about all available flags, add `-h` or `--help` to the command above.

## Tracking/Prediction on 3D

If you are not only interested in detecting spines on single 2D images but in a 3D stack consisting of multiple images, you can use the tracking command as follows:

```bash
python src/spine_detection/tracking_mmdet.py \
    --input "data/raw/person1/SR052N1D1day1*.png"\
    --model Cascade_RCNN \
    --param_config lr_0.001_warmup_None_momentum_0.6_L2_3e-06_run_1 \
    --model_epoch latest \
    --save_images
```

The basic parameters are the same as with the `predict_mmdet.py` script. However, you have more options to customize the tracker, e.g. which metric should be used for box overlap between different images.

Similarly, the output can be found by default under `output/tracking/<model>/<param_config>/` with a folder `images` and a file `data_tracking_<model-type>_aug_<use_aug>_<model_epoch>_theta_<theta>_delta_<delta>_<input_mode>.csv`. All these parameters can be explored by looking into the code or by running the tracking file with the `-h` or `--help` flag.

### Evaluate tracking

After having tracked the dendritic spines over 3D stacks, you can evaluate the models performance when the groundtruth is also available for the analyzed data as follows:

```bash
python src/spine_detection/evaluate_tracking_mmdet.py \
    --model Cascade_RCNN \
    --param_config lr_0.001_warmup_None_momentum_0.6_L2_3e-06_run_1 \
    --tracking data_tracking_default_aug_False_epoch_1_theta_0.5_delta_0.5_Test.csv \
    --gt_file output/tracking/GT/data_tracking_max_wo_offset.csv
```

For the `tracking` flag you should choose the name of the `data_tracking_<custom-name>.csv` file generated in the above section about [tracking](#trackingprediction-on-3d). The `gt_file` argument should correspond to a csv file consisting of tracked labeled data. Unfortunately, simply creating such a file of tracked labeled data is not yet available.

## Training

For every kind of training you need to prepare your data correctly. Your data should be in the same format as the data you could download from us, especially the labels and the folder structure. If you want to create and use your own dataset, save the train, val and test labels somewhere inside the `data` folder together with your images. It is not important if these are in the same folder as the labels but the path in the labels should be correct and start with `data/`.

After having downloaded our custom models you can continue training them with your own data with the following command:

```bash
python src/spine_detection/train_mmdet.py \
    --model Cascade_RCNN_model \
    --model_type Cascade-RCNN \
    --checkpoint lr_0.001_warmup_None_momentum_0.6_L2_3e-06_run_1/epoch_10.pth \
    --resume
```

In that case the script is searching for a model file (`.pth`) inside `tutorial_exps/<model_name>/<checkpoint>`. If the `model` flag is provided, `model_name=model`. Otherwise `model_name` is the value of `model_paths.<model_type>.base_checkpoint` in the config file `configs/model_config_paths.yaml` appended by `no_data_augmentation` or `data_augmentation` depending if the `--use_aug` flag is added (see the instructions for [prediction](#predictioninference-on-2d) as well). However, if you have downloaded our data as described in the [installation](#installation), the models should already have the correct paths.

The `resume` flag ensures that the training with the already existing model is continued and not started from scratch again.

### Train on pretrained model from mmdetection modelzoo

If you want to train on any pretrained model you first need to choose which pretrained model you want to use and download it into `references/mmdetection/checkpoint`. You now have downloaded a model with name `<model_type>_<param_config>_<date>-<identifier>.pth`. If you can find any `model_paths` with `base_config` equal to `<model>_<param_config>_coco` in `configs/model_config_paths.yaml` you are good to go.

Otherwise the procedure is a bit more complex:

First you need to create a new config file `cfg_file=references/mmdetection/configs/<model>/<model>_<param_config>_coco_SPINE.py`. The config file should look like the the corresponding python file without the `SPINE` addition.

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

where `<custom_model_type>` can be any custom name you would like to use.

<br />

After preparing the data and the config files, the training can then be started with

```bash
python src/spine_detection/train_mmdet.py \
    --model_type <custom_model_type>
```

More command line parameters as the number of epochs or the learning rate are available. A list of all parameters can be found in the code or by using the `-h` or `--help` flag.

## FaQ

**Q: There are no detections and I get the warning `missing key in source state_dict: ...` or `unexpected key in source state_dict: ...`**

**A:** This means the model weights from the `.pth` file cannot be correctly matched to the expected model weights. This probably happens because the `--model_type` flag does not fit for the loaded model weights. The config file `configs/model_config_paths.yaml` shows which model type should be entered so that the correct weights are expected. For more details see [here](#predictioninference).

**Q: The installation is stuck in the `Building wheel for mmcv-full (setup.py) ...` phase**

**A:** For certain python versions there is no pre-compiled version of `mmcv` available, so it needs to be downloaded, compiled and installed. This may take more than ten minutes, so please be patient.
