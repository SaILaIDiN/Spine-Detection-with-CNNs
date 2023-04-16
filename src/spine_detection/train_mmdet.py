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
from spine_detection.utils.logger_utils import setup_custom_logger
from spine_detection.utils.model_utils import (
    get_checkpoint_path,
    get_config_path,
    get_pretrained_checkpoint_path,
    load_config,
    parse_args,
)

logger = logging.getLogger(__name__)


def train_main(args):
    model_type = args.model_type
    model_suffix = args.model_suffix
    use_aug = args.use_aug

    cfg_path = pkg_resources.resource_filename("spine_detection", "configs/model_config_paths.yaml")
    paths_cfg = load_config(cfg_path)

    model_folder = "tutorial_exps"

    if model_type is None:
        model_type = "default"

    if args.model is None:
        dir_train_checkpoint = get_checkpoint_path(model_type, model_folder, use_aug, paths_cfg)
    else:
        dir_train_checkpoint = f"{model_folder}/{args.model}"
    config_file = get_config_path(model_type, use_aug, paths_cfg)
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
    if args.resume:
        if not args.checkpoint:
            args.checkpoint = (
                "lr_" + str(args.learning_rate) + "_warmup_" + str(args.warm_up) + "_momentum_" + str(args.momentum)
            )

        cfg.resume_from = os.path.join(dir_train_checkpoint, args.checkpoint)
    else:
        # # Define the correct checkpoint file to start the model training with pretrained weights
        coco_checkpoint = get_pretrained_checkpoint_path(model_type, paths_cfg, model_suffix)
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
    args = parse_args(mode="train")
    logger = setup_custom_logger(__name__, args.log_level)
    logger.debug(f"Args: {args}")
    train_main(args)
