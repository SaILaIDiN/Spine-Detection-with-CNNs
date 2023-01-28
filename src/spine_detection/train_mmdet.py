import argparse
import copy
import logging
import os
import os.path as osp

import mmcv
import pkg_resources
from mmcv import Config
from mmdet.apis.train import set_random_seed, train_detector
from mmdet.datasets.builder import build_dataset
from mmdet.models.builder import build_detector

from spine_detection.utils.data_utils import DATASETS, SpineDataset
from spine_detection.utils.model_utils import (
    get_checkpoint_path,
    get_config_path,
    get_pretrained_checkpoint_path,
    load_config,
)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Train a model with given config and checkpoint file",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("-tr", "--train_csv", default=None, help="annotation file for training data")
parser.add_argument("-sp", "--special_term", default="", help="name appendix to store in different train folders")
parser.add_argument(
    "-mt",
    "--model_type",
    help="decide which model to use as config and checkpoint file. " "use one of [Cascade_RCNN, GFL, VFNet, Def_DETR]",
)
parser.add_argument("-ms", "--model_suffix", help="Suffix of checkpoint model, usually starting with the date.")
parser.add_argument(
    "-ua", "--use_aug", action="store_true", help="decide to load the config file with or without data augmentation"
)
parser.add_argument("-sd", "--seed_data", default=0, help="seed for the data loader, random if None")
parser.add_argument("-sw", "--seed_weights", default=0, help="seed for initial random weights")
# The following arguments are primary parameters for optimization
# if there is no default defined, this parameter will take up the predefined value in the config file
parser.add_argument("-lr", "--learning_rate", default=0.0005)
parser.add_argument("-me", "--max_epochs", default=10)
parser.add_argument("-wu", "--warm_up", default=None, help="learning rate warm up, use None to disable")
parser.add_argument("-st", "--steps_decay", default=None, help="steps for lr decay")
parser.add_argument("-mom", "--momentum", default=None, help="only for optimizer SGD")
parser.add_argument("-wd", "--weight_decay", default=None, help="only for optimizer SGD")
parser.add_argument("-do", "--dropout", default=None, help="overloading this parameter, varies by model type!")

# The following arguments are secondary parameters used for data augmentation
parser.add_argument("-prbc", "--p_rbc", default=None, help="p for RandomBrightnessContrast")
parser.add_argument("-rb", "--random_brightness", default=None, help="from RandomBrightnessContrast")
parser.add_argument("-rc", "--random_contrast", default=None, help="from RandomBrightnessContrast")
parser.add_argument("-vf", "--vertical_flip", default=None, help="p for VerticalFlip")
parser.add_argument("-hf", "--horizontal_flip", default=None, help="p for HorizontalFlip")
parser.add_argument("-ro", "--rotate", default=None, help="p for Rotate")

# The following arguments are secondary parameters used for the anchor generator in RCNN-models
# # # MISSING


def train_main(args):

    model_type = args.model_type
    model_suffix = args.model_suffix
    use_aug = args.use_aug

    cfg_path = pkg_resources.resource_filename("spine_detection", "configs/model_config_paths.yaml")
    paths_cfg = load_config(cfg_path)

    model_folder = "tutorial_exps"

    if model_type is None:
        model_type = "default"
    dir_train_checkpoint = get_checkpoint_path(model_type, model_folder, use_aug, paths_cfg)
    config_file = get_config_path(model_type, use_aug, paths_cfg)
    coco_checkpoint = get_pretrained_checkpoint_path(model_type, paths_cfg, model_suffix)
    logger.info("Loading model ...")

    # # # Set up config file to manipulate for training
    cfg = config_file
    if args.train_csv is not None:
        cfg.data.train.ann_file = args.train_csv
    cfg.seed = args.seed_data
    if args.seed_weights is not None:
        set_random_seed(args.seed_weights, deterministic=False)
    cfg.gpu_ids = range(1)

    # # # NOTE: either use 'cfg.resume_from' or 'cfg.load_from'
    # this loads the pretrained weights on COCO dataset into our model (or resume from a model)
    # cfg.resume_from = os.path.join(dir_train_checkpoint,
    #                                'lr_' + str(args.learning_rate) + '_warmup_' + str(args.warm_up) +
    #                                '_momentum_' + str(args.momentum))

    # # Define the correct checkpoint file to start the model training with pretrained weights
    cfg.load_from = coco_checkpoint

    # directory for the trained model weights
    cfg.work_dir = os.path.join(
        dir_train_checkpoint,
        "lr_"
        + str(args.learning_rate)
        + "_warmup_"
        + str(args.warm_up)
        + "_momentum_"
        + str(args.momentum)
        + "_L2_"
        + str(args.weight_decay)
        + str(args.special_term),
    )

    # # # NOTE: the usage of 'if args.XYZ is not None:' means that if the parser passes a value of type None,
    # the config file will not be updated inside train_mmdet.py and thus keeps its default config of that feature!
    # So be sure about which parameter/feature needs this or not.

    cfg.workflow = [("train", 1), ("val", 1)]
    cfg.optimizer.lr = args.learning_rate
    cfg.lr_config.warmup = args.warm_up
    if args.steps_decay is not None:
        cfg.lr_config.step = args.steps_decay
    cfg.runner.max_epochs = args.max_epochs

    if args.momentum is not None:
        cfg.optimizer.momentum = args.momentum
    if args.weight_decay is not None:
        cfg.optimizer.weight_decay = args.weight_decay

    if args.model_type == "Def_DETR" and args.dropout is not None:
        cfg.model.bbox_head.transformer.encoder.transformerlayers.ffn_dropout = args.dropout
        cfg.model.bbox_head.transformer.decoder.transformerlayers.attn_cfgs[0].dropout = args.dropout
        cfg.model.bbox_head.transformer.decoder.transformerlayers.ffn_dropout = args.dropout

    # # Config adjustment for Data Augmentation
    if args.p_rbc is not None:
        cfg.albu_train_transforms[3].p = args.p_rbc
    if args.random_brightness is not None:
        cfg.albu_train_transforms[3].brightness_limit = args.random_brightness
    if args.random_contrast is not None:
        cfg.albu_train_transforms[3].contrast_limit = args.random_contrast
    if args.vertical_flip is not None:
        cfg.albu_train_transforms[0].p = args.vertical_flip
    if args.horizontal_flip is not None:
        cfg.albu_train_transforms[1].p = args.horizontal_flip
    if args.rotate is not None:
        cfg.albu_train_transforms[2].p = args.rotate

    # We can initialize the logger for training and have a look
    # at the final config used for training
    logger.info(f"Config:\n{cfg.pretty_text}")

    cfg.device = "cuda"
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    # Build the detector
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # Train the detector
    train_detector(model, datasets, cfg, distributed=False, validate=True)
    return cfg.work_dir


if __name__ == "__main__":
    args = parser.parse_args()
    train_main(args)
