from mmdet.apis import inference_detector, init_detector, show_result_pyplot

config = "configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py"
checkpoint = "checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth"
model = init_detector(config, checkpoint, device="cuda:0")
