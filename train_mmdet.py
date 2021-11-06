import argparse
import os
import mmcv
from mmcv import Config
from mmdet.apis.train import set_random_seed
from dataset_mmdet import SpineDataset, DATASETS
import os.path as osp
from mmdet.datasets.builder import build_dataset
from mmdet.models.builder import build_detector
from mmdet.apis.train import train_detector

parser = argparse.ArgumentParser(description='Train a model with given config and checkpoint file',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-mt', '--model_type',
                    help='decide which model to use as config and checkpoint file. '
                         'use one of [Cascade_RCNN, GFL, VFNet, Def_DETR]')
parser.add_argument('-ua', '--use_aug', default='False',
                    help='decide to load the config file with or without data augmentation')
# The following arguments are primary parameters for optimization
# if there is no default defined, this parameter will take up the predefined value in the config file
parser.add_argument('-lr', '--learning_rate', default=0.0005)
parser.add_argument('-me', '--max_epochs', default=10)
parser.add_argument('-wu', '--warm_up', default=None, help='learning rate warm up, use None to disable')
parser.add_argument('-st', '--steps_decay', default=None, help='steps for lr decay')
parser.add_argument('-mom', '--momentum', default=None, help='only for optimizer SGD')
parser.add_argument('-wd', '--weight_decay', default=None, help='only for optimizer SGD')
parser.add_argument('-do', '--dropout', default=None, help='overloading this parameter, varies by model type!')

# The following arguments are secondary parameters used for data augmentation
parser.add_argument('-rb', '--random_brightness', default=None, help='from RandomBrightnessContrast')
parser.add_argument('-rc', '--random_contrast', default=None, help='from RandomBrightnessContrast')
parser.add_argument('-vf', '--vertical_flip', default=None, help='p for VerticalFlip')
parser.add_argument('-hf', '--horizontal_flip', default=None, help='p for HorizontalFlip')
parser.add_argument('-ro', '--rotate', default=None, help='p for Rotate')

# The following arguments are secondary parameters used for the anchor generator in RCNN-models
# # # MISSING


def train_main(args):

    model_type = args.model_type
    use_aug = args.use_aug

    model_folder = "tutorial_exps"
    print("[INFO] Loading model ...")
    if model_type == "Cascade-RCNN":
        if use_aug == "True":
            dir_train_checkpoint = os.path.join(model_folder, "Cascade_RCNN_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco_SPINE_AUG.py")
        else:
            dir_train_checkpoint = os.path.join(model_folder, "Cascade_RCNN_no_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco_SPINE.py")
        coco_checkpoint = \
            'references/mmdetection/checkpoints/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth'
    elif model_type == "GFL":
        if use_aug == "True:":
            dir_train_checkpoint = os.path.join(model_folder, "GFL_RX101_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_SPINE_AUG.py")
        else:
            dir_train_checkpoint = os.path.join(model_folder, "GFL_RX101_no_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_SPINE.py")
        coco_checkpoint = 'references/mmdetection/checkpoints/' \
                          'gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth'
    elif model_type == "VFNet":
        if use_aug == "True":
            dir_train_checkpoint = os.path.join(model_folder, "VFNet_RX101_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_SPINE_AUG.py")
        else:
            dir_train_checkpoint = os.path.join(model_folder, "VFNet_RX101_no_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_SPINE.py")
        coco_checkpoint = 'references/mmdetection/checkpoints/' \
                          'vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth'
    elif model_type == "Def_DETR":
        dir_train_checkpoint = os.path.join(model_folder, "Def_DETR_R50_no_data_augmentation")
        config_file = Config.fromfile(
            "references/mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco_SPINE.py")
        coco_checkpoint = 'references/mmdetection/checkpoints/' \
                          'deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'
    else:
        dir_train_checkpoint = os.path.join(model_folder, "Cascade_RCNN")
        config_file = Config.fromfile(
            "references/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco_SPINE.py")
        coco_checkpoint = 'references/mmdetection/checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'

    # # # Set up config file to manipulate for training
    cfg = config_file

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # this loads the pretrained weights on COCO dataset into our model (or resume from a model)
    # cfg.resume_from = 'tutorial_exps/epoch_14.pth'

    # # Define the correct checkpoint file to start the model training with pretrained weights
    cfg.load_from = coco_checkpoint

    # directory for the trained model weights
    cfg.work_dir = os.path.join(dir_train_checkpoint,
                                'lr_' + str(args.learning_rate) + '_warmup_' + str(args.warm_up))

    # # # NOTE: the usage of 'if args.XYZ is not None:' means that if the parser passes a value of type None,
    # the config file will not be updated inside train_mmdet.py and thus keeps its default config of that feature!
    # So be sure about which parameter/feature needs this or not.
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
        # cfg.model.bbox_head.transformer.decoder.transformerlayers.attn_cfgs.dropout = args.dropout
        cfg.model.bbox_head.transformer.decoder.transformerlayers.ffn_dropout = args.dropout

    # # Config adjustment for Data Augmentation
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
    print(f'Config:\n{cfg.pretty_text}')

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # Train the detector
    train_detector(model, datasets, cfg, distributed=False, validate=True)
    return


if __name__ == "__main__":
    args = parser.parse_args()
