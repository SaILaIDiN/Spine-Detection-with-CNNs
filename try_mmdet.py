from mmdet.apis import init_detector, inference_detector


def check_inference(config_file, checkpoint_file, image):
    """ Simple function to check inference on single image and visualize result """

    device = 'cuda:0'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # define spine image
    # inference of the demo image
    result = inference_detector(model, image)
    # save the visualization results to image files
    model.show_result(image, result, score_thr=0.3, font_size=6, out_file='result.png')
    # k = result
    # print("TYPE: ", type(k), type(k[0][0]), type(k[0][1]), type(k[0][2]), len(k), len(k[0]), "\nKKKKK:", k)


if __name__ == "__main__":
    from mmcv import Config

    cfg = Config.fromfile("references/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco_SPINE.py")
    checkpoint = "tutorial_exps/epoch_4_no_shuffle_055mAP.pth"
    img = "data/raw/person1/SR052N1D2day1stack3-17.png"

    check_inference(cfg, checkpoint, img)
