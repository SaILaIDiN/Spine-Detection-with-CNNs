import mmcv
from mmcv import Config
from mmdet.apis.train import set_random_seed
from dataset_mmdet import SpineDataset, DATASETS
# # CASCADE-RCNN
# cfg = Config.fromfile("references/mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco_SPINE.py")
# cfg = Config.fromfile("references/mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco_SPINE_AUG.py")
# cfg = Config.fromfile("references/mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco_SPINE.py")
# cfg = Config.fromfile("references/mmdetection/configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco_SPINE.py")
# cfg = Config.fromfile("references/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco_SPINE.py")
# cfg = Config.fromfile("references/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco_SPINE_AUG.py")

# # GENERALIZED FOCAL LOSS
# cfg = Config.fromfile("references/mmdetection/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_SPINE.py")
# cfg = Config.fromfile("references/mmdetection/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_SPINE_AUG.py")
# cfg = Config.fromfile("references/mmdetection/configs/gfl/gfl_x101_32x4d_fpn_mstrain_2x_coco_SPINE.py")
# cfg = Config.fromfile("references/mmdetection/configs/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_SPINE.py")

# # VARIFOCALNET
# cfg = Config.fromfile("references/mmdetection/configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_SPINE.py")
# cfg = Config.fromfile("references/mmdetection/configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_SPINE_AUG.py")

# # DEFORMABLE DETR
cfg = Config.fromfile("references/mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco_SPINE.py")
# cfg = Config.fromfile("references/mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco_SPINE_AUG.py")

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# this loads the pretrained weights on COCO dataset into our model (or resume from a model)
cfg.resume_from = 'tutorial_exps/epoch_14.pth'

# # CASCADE-RCNN
# cfg.load_from = 'references/mmdetection/checkpoints/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth'
# cfg.load_from = 'references/mmdetection/checkpoints/cascade_rcnn_x101_32x4d_fpn_1x_coco_20200316-95c2deb6.pth'
# cfg.load_from = 'references/mmdetection/checkpoints/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth'
# cfg.load_from = 'references/mmdetection/checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'

# # GENERALIZED FOCAL LOSS
# cfg.load_from = 'references/mmdetection/checkpoints/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth'
# cfg.load_from = 'references/mmdetection/checkpoints/gfl_x101_32x4d_fpn_mstrain_2x_coco_20200630_102002-50c1ffdb.pth'
# cfg.load_from = 'references/mmdetection/checkpoints/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth'

# # VARIFOCALNET
# cfg.load_from = 'references/mmdetection/checkpoints/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth'

# # DEFORMABLE DETR
# cfg.load_from = 'references/mmdetection/checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth'


# directory for the trained model weights
cfg.work_dir = 'tutorial_exps'

cfg.optimizer.lr = 0.00005
cfg.lr_config.warmup = None
# cfg.lr_config.warmup = "linear"
cfg.runner.max_epochs = 15

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


if __name__ == "__main__":
    import os.path as osp
    from mmdet.datasets.builder import build_dataset
    from mmdet.models.builder import build_detector
    from mmdet.apis.train import train_detector

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
