import mmcv
from mmcv import Config
from mmdet.apis.train import set_random_seed
from dataset_mmdet import SpineDataset, DATASETS

cfg = Config.fromfile("references/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco_SPINE.py")

# # These are the main changes in a config file to fit our single class spine dataset
# cfg.dataset_type = "SpineDataset"
# cfg.data_root = "data/raw/"
# cfg.img_norm_cfg.to_rgb = True

# cfg.train_pipeline[2].img_scale = (512, 512)
# cfg.test_pipeline[1].img_scale = (512, 512)
#
# cfg.data.train.pipeline[2].img_scale = (512, 512)
# cfg.data.test.pipeline[1].img_scale = (512, 512)
# cfg.data.val.pipeline[1].img_scale = (512, 512)

# cfg.data.samples_per_gpu = 1
# cfg.data.workers_per_gpu = 1
#
# cfg.data.train.type = "SpineDataset"
# cfg.data.train.ann_file = "data/default_annotations/data_train.csv"
# cfg.data.train.img_prefix = ""
#
# cfg.data.val.type = "SpineDataset"
# cfg.data.val.ann_file = "data/default_annotations/data_val.csv"
# cfg.data.val.img_prefix = ""
#
# cfg.data.test.type = "SpineDataset"
# cfg.data.test.ann_file = "data/default_annotations/data_val.csv"
# cfg.data.test.img_prefix = ""
#
# cfg.evaluation.interval = 1
# cfg.evaluation.metric = 'mAP'
#
# cfg.model.roi_head.bbox_head[0].num_classes = 1
# cfg.model.roi_head.bbox_head[1].num_classes = 1
# cfg.model.roi_head.bbox_head[2].num_classes = 1

cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# this loads the pretrained weights on COCO dataset into our model
cfg.load_from = 'references/mmdetection/checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
# directory for the trained model weights
cfg.work_dir = 'tutorial_exps'

# cfg.optimizer.lr = 0.00002
cfg.lr_config.warmup = None
cfg.runner.max_epochs = 10

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
