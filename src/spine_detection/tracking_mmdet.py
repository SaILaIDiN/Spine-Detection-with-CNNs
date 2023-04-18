import argparse
import glob
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm

from spine_detection.predict_mmdet import predict_images
from spine_detection.utils.data_utils import csv_to_boxes
from spine_detection.utils.logger_utils import setup_custom_logger
from spine_detection.utils.model_utils import load_model, parse_args
from spine_detection.utils.opencv_utils import draw_boxes, image_load_encode
from spine_detection.utils.tracker import CentroidTracker as CT

logger = logging.getLogger(__name__)


def tracking_main(args):
    # Max diff -> (minimum) diff so that two following bboxes are connected with each other
    # iom thresh -> min iom that two boxes are considered the same in the same frame!
    MAX_DIFF = args.tau
    IOM_THRESH = args.theta
    THRESH = args.delta  # This threshold parameter is the same one used in predict.py's predict_images()
    MIN_APP = args.appeared
    MAX_DIS = args.disappeared
    METRIC = args.metric
    NUM_CLASSES = 1
    MAX_VOL = 2000

    if args.images is None:
        raise ValueError("You need to specify input images or input tif stack!")

    # save_folder: folder where tracking csv file will be saved
    # folder: name of folder which is used in csv file for generating filename-column
    model_name = (
        args.model_type
    )  # args.model.split("/")[-1] if args.model.split("/")[-1] != "" else args.model.split("/")[-2]
    if args.output is None:
        args.output = os.path.join("output/tracking/", model_name, args.param_config)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    img_output_path = os.path.join(args.output, "images")
    csv_output_path = args.output
    Path(csv_output_path).mkdir(parents=True, exist_ok=True)
    if args.model_type is None:
        args.model_type = "default"
    # cfg_path = pkg_resources.resource_filename("spine_detection", "configs/model_config_paths.yaml")
    # paths_cfg = load_config(cfg_path)

    # model_folder = "tutorial_exps"

    # if model_type is None:
    #     model_type = "default"
    # dir_train_checkpoint = get_checkpoint_path(model_type, model_folder, use_aug, paths_cfg)
    # config_file = get_config_path(model_type, use_aug, paths_cfg)
    # coco_checkpoint = get_pretrained_checkpoint_path(model_type, paths_cfg, model_suffix)

    csv_output_path = os.path.join(
        csv_output_path,
        args.file_save
        + "_"
        + args.model_type
        + "_aug_"
        + str(args.use_aug)
        + "_"
        + args.model_epoch
        + "_theta_"
        + str(args.theta)
        + "_delta_"
        + str(args.delta)
        + "_"
        + args.input_mode
        + ".csv",
    )
    if args.save_images and not os.path.exists(img_output_path):
        os.makedirs(img_output_path)

    # to get some annotations on the first images too, make the same backwards
    if args.input_mode == "Test":
        all_imgs = sorted(glob.glob(args.images))
    elif args.input_mode == "Train" or args.input_mode == "Val":
        df = pd.read_csv(args.images)
        all_imgs = df["filename"].tolist()
        all_imgs = list(dict.fromkeys(all_imgs))
    else:
        all_imgs = []
        logger.error("Wrong input mode!")

    all_dicts = []
    total_boxes = []
    total_classes = []
    total_scores = []
    nr_imgs = len(list(all_imgs))
    objects = dict()

    # if it's just a single csv file, load all data before iterating over images
    if args.csv is not None:
        all_csv_files = glob.glob(args.csv)
        if len(all_csv_files) == 0:
            raise ValueError("No csv files with valid prediction data are available.")
        csv_path = args.csv

    # get all boxes, scores and classes at the start if prediction is necessary:
    if args.csv is None:
        model = load_model(args.model_type, args.use_aug, args.model_epoch, args.param_config, device=args.device)
        # We currently disable storing prediction images for tracking. Replace None by img_output_path to activate
        # Other way of disabling is complicated in here and coupled to args.save_images which is always False here?!
        all_boxes, all_scores, all_classes, all_num_detections = predict_images(
            model,
            args.images,
            None,
            csv_output_path,
            theta=IOM_THRESH,
            delta=THRESH,
            save_csv=False,
            return_csv=True,
            input_mode=args.input_mode,
        )

    all_csv_paths = list(Path().rglob(args.csv))

    ct = CT(
        maxDisappeared=MAX_DIS,
        minAppeared=MIN_APP,
        maxDiff=MAX_DIFF,
        iomThresh=IOM_THRESH,
        maxVol=MAX_VOL,
        metric=METRIC,
    )

    # get offsets if we want to use them
    if args.use_offsets == "True":
        sr, neuron, dend, day = 52, 1, 1, 1
        arrx = scipy.io.loadmat(f"data/offsets/SR{sr}N{neuron}D{dend}offsetX.mat")[f"SR{sr}N{neuron}D{dend}offsetX"]
        arry = scipy.io.loadmat(f"data/offsets/SR{sr}N{neuron}D{dend}offsetY.mat")[f"SR{sr}N{neuron}D{dend}offsetY"]

        # get offset for each stack
        offsets = np.array(list(zip(arrx[:, day - 1], arry[:, day - 1]))).astype(int)

        # double offsets so that it can easily be added to bounding boxes
        offsets = np.concatenate((offsets, offsets), axis=1)  # Sized in multiples of 512 == weight == height

        # make offset positive by subtracting possible negative offsets (including first offset of 0)
        offsets = offsets - np.min(offsets, axis=0)

    # use given prediction for all images, if csv is available
    for i, img in tqdm(enumerate(all_imgs), total=len(all_imgs)):
        orig_img = Path(img).name
        if args.csv is not None:  # PART with no new prediction, instead use csv output from previous prediction
            # NOTE: make sure, that the used csv files are from the correct prediction model/ pth file!
            if len(all_csv_paths) > 1:
                csv_path = [elem for elem in all_csv_paths if orig_img[:-4] == elem.name[:-4]]
                if len(csv_path) == 0:
                    # no corresponding csv file for this image
                    continue
                else:
                    csv_path = csv_path[0]
                try:
                    new_df = pd.read_csv(csv_path)
                    boxes, scores, classes, num_detections = csv_to_boxes(new_df)
                    boxes = np.asarray(boxes)
                    scores = np.asarray(scores)
                except:
                    continue
            else:  # Case for single csv-file, here meaning single image
                try:
                    new_df = pd.read_csv(args.csv)

                    # load only data from interesting image
                    new_df = new_df[new_df.apply(lambda row: os.path.splitext(orig_img)[0] in row["filename"], axis=1)]
                    # axis=1 for looping through rows, to remove the '.png' extension in the filename
                    boxes, scores, classes, num_detections = csv_to_boxes(new_df)
                    boxes = np.asarray(boxes)
                    scores = np.asarray(scores)
                except:
                    continue
        else:
            # just load data from saved list
            # this works as all_imgs from this file and sorted(glob.glob(args.images)) from predict sort all
            # image paths so they are perfectly aligned
            # NOTE: the output values from the prediction are of type np.ndarray
            boxes, scores, classes, num_detections = all_boxes[i], all_scores[i], all_classes[i], all_num_detections[i]

        if args.csv is not None:
            boxes = boxes[0]

        # look if there are some boxes
        if len(boxes) == 0:
            # print("NO BOXES!")
            continue
        # else:
        #     print("BOXES!")

        # print("BOXES: ", boxes, type(boxes), type(boxes[0]))
        # print("SCORES: ", scores, type(scores))
        # print("NUM-DETECTIONS: ", num_detections, type(num_detections))
        image_np, orig_w, orig_h = image_load_encode(img)
        h, w = image_np.shape[:2]

        # # # Real tracking part!
        # the if-condition with THRESH manages both origins of boxes, scores and num_detections to become equal now
        # this is because the prediction output delivers the unfiltered number of detections,
        # while the csv created in predict_images() is already filtered by the same threshold as THRESH!

        if args.csv is not None:
            scores = scores[0]
            num_detections = num_detections[0]

        # convert all detections from different stacks into one stack (via offset matlab files)
        if args.use_offsets == "True":
            # format of img name: SR52N1D1day1stack1-xx.png
            stack_nr = int(orig_img[-8])
            # print("BOXES shape: ", boxes.shape)
            # print("OFFSETS shape: ", offsets[stack_nr - 1].shape)
            boxes += offsets[stack_nr - 1]

        rects = np.array(
            [
                [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i]]
                for i in range(num_detections)
                if scores[i] >= THRESH
            ]
        )

        # print("RECTS: ", rects, rects.shape)
        objects = ct.update(rects)  # y1,x1,y2,x2

        # Start with non-empty lists
        boxes = []
        classes = []
        scores = []

        # DO NOT USE absolute path for images!
        total_path = os.path.join(img_output_path, img.split("/")[-1])
        for key in objects:
            orig_dict = {"filename": total_path, "width": w, "height": h, "class": "spine"}

            # Making boxes, classes, scores correct
            cX, cY, width, height, conf = objects[key]
            x1, x2 = (cX - width / 2) / w, (cX + width / 2) / w
            y1, y2 = (cY - height / 2) / h, (cY + height / 2) / h
            boxes.append([x1, y1, x2, y2])
            classes.append(1)
            scores.append(conf)

            orig_dict.update(
                {
                    "id": key,
                    "ymin": round(y1 * h, 2),
                    "ymax": round(y2 * h, 2),
                    "xmin": round(x1 * w, 2),
                    "xmax": round(x2 * w, 2),
                    "score": conf,
                }
            )

            all_dicts.append(orig_dict)

        boxes = np.array(boxes)
        classes = np.array(classes)
        scores = np.array(scores)
        total_boxes.append(boxes)
        total_classes.append(classes)
        total_scores.append(scores)

        if args.save_images:
            image_np = cv2.imread(img)
            image_np = draw_boxes(image_np, objects)
            cv2.imwrite(total_path, image_np)

    # delete all double elements
    all_dicts = [dict(tup) for tup in {tuple(set(elem.items())) for elem in all_dicts}]
    df = pd.DataFrame(
        all_dicts, columns=["id", "filename", "width", "height", "class", "score", "xmin", "ymin", "xmax", "ymax"]
    )
    df.sort_values(by="filename", inplace=True)
    df.to_csv(csv_output_path, index=False)

    # count real spines (does NOT correspond to max_key, but to number of keys!)
    nr_all_ind = len(df.groupby("id"))
    logger.info(f"Nr of spines found: {nr_all_ind}")

    logger.info("Written predictions to " + csv_output_path + ".")
    return


if __name__ == "__main__":
    args = parse_args(mode="tracking")
    logger = setup_custom_logger(__name__, args.log_level)
    logger.debug(f"Args: {args}")
    tracking_main(args)
