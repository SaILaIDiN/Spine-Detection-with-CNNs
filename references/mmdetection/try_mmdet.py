from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
k = inference_detector(model, 'demo/demo.jpg')
# k_2 = model("demo/demo.jpg")
print("TYPE: ", type(k), type(k[0][0]), type(k[0][1]), type(k[0][2]), len(k), len(k[0]), "\nKKKKK:", k)

# print("TYPE: ", type(k_2), type(k_2[0][0]), type(k_2[0][1]), type(k_2[0][2]), len(k_2), len(k_2[0]), k_2)
