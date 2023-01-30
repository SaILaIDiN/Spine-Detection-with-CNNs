import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
import cv2
import numpy as np
import pandas as pd
import pkg_resources
from mmcv import Config
from mmdet.apis import inference_detector, init_detector
from tqdm import tqdm

from spine_detection.utils.data_utils import calc_metric_xy
from spine_detection.utils.model_utils import (
    get_checkpoint_path,
    get_config_path,
    load_config,
    load_model,
    parse_args,
)
from spine_detection.utils.opencv_utils import (
    draw_boxes_predict,
    image_decode,
    image_load_encode,
)

sys.path.append("..")


def postprocess(boxes, scores, theta=0.5):
    """Postprocess boxes and scores and average boxes if necessary
    Args:
        boxes (np.ndarray): input boxes in (x1, y1, x2, y2) format
        scores (np.ndarray): confidence scores
        theta (float, optional): minimum IoM thresh to count as same object. Defaults to 0.5.
    Returns:
        Tuple[np.ndarray]: tuple of correct np arrays (boxes, scores)
    """
    # postprocess boxes and scores:
    # if multiple boxes have an iom >= theta -> consider as the same box and get
    # expected averaged box out of it
    final_boxes = []
    final_scores = []
    cluster_ids = list(range(len(boxes)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iom = calc_metric_xy(rect1=boxes[i], rect2=boxes[j])
            if iom >= theta:
                # set cluster id of following point to that of the current pnt
                # this order is correct, as that is already fixed from previous rounds
                cluster_ids[j] = cluster_ids[i]

    cluster_ids = np.array(cluster_ids)
    # for all clusters calculate an average box
    for cluster_id in sorted(set(cluster_ids)):
        # get all indices with that cluster id
        indices = np.where(cluster_ids == cluster_id)[0]

        # only average if there are more than one box in that cluster
        if len(indices) > 1:
            new_box = np.sum(boxes[indices] * scores[indices].reshape(len(indices), 1), axis=0) / np.sum(
                scores[indices]
            )
            max_score = np.max(scores[indices])

            # score calculates as follows: max_score + weight*extra, weight = 1-max_score (to stay <= 1)
            # extra = sum(weight_i * score_i) with weight_i = score_i/(sum(scores) - max_score)
            # extra is <= 1 as well, it corresponds to the average of scores without max_score
            new_score = max_score + (1 - max_score) * (np.sum(scores[indices] ** 2) - max_score**2) / (
                np.sum(scores[indices]) - max_score
            )
            final_boxes.append(new_box.astype(np.int64))
            final_scores.append(new_score)
        else:
            final_boxes.append(boxes[indices[0]])
            final_scores.append(scores[indices[0]])
    return np.array(final_boxes), np.array(final_scores)


def write_to_df(df, img_path, w, h, csv_path, class_label, boxes, scores, thresh=0.0, disable_thresh=False):
    """Write detection to dataframe
    Args:
        df (pd.DataFrame): dataframe which should be appended
        img_path (str): image path of image corresponding to detections
        w (int): width of image
        h (int): height of image
        csv_path (str): path to folder where all csv files should be saved
        class_label (str): name of class
        boxes (np.ndarray): all detection boxes
        scores (np.ndarray): all detection scores
        thresh (float, optional): min confidence necessary to count as spine. Defaults to 0.0.
        disable_thresh (bool, optional): Flag whether to use differentiation by confidence score. Defaults to False.
    Returns:
        pd.DataFrame: appended dataframe
    """
    # 'filename', 'width', 'height', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax'
    # boxes are in format: [y1, x1, y2, x2] between 0 and 1 !!!!
    dict_list = []
    for i in range(len(boxes)):
        if not disable_thresh and scores[i] < thresh:
            continue
        box = image_decode(rect=boxes[i])
        # box = boxes[i]
        dict_list.append(
            {
                "filename": img_path,
                "width": w,
                "height": h,
                "class": class_label,
                "score": scores[i],
                "xmin": box[0],
                "ymin": box[1],
                "xmax": box[2],
                "ymax": box[3],
            }
        )
    if len(dict_list) != 0:
        df = pd.concat([df, pd.DataFrame(dict_list)])
        # df = df.append(dict_list)
    # be aware of windows adding \\ for folders in paths!
    csv_filepath = os.path.join(csv_path, Path(img_path).name[:-4] + ".csv")
    df.to_csv(csv_filepath, index=False)
    logger.info("Detections saved in csv file " + csv_filepath + ".")
    return df


def df_to_data(df: pd.DataFrame):
    """Converts GT dataframe to detections and their classes with confidences 1.0
    Args:
        df (pd.DataFrame): input dataframe for GT
    Returns:
        Tuple[List]: tuple of detection rects, detection classes
    """
    # get rects (boxes+scores) and classes for this specific dataframe
    rects = np.zeros((len(df), 5))
    scores = np.zeros(len(df))

    if len(df) == 0:
        return rects, scores
    fi = df.first_valid_index()
    w, h = df["width"][fi], df["height"][fi]
    for i in range(len(df)):
        rects[i] = np.array([df["xmin"][fi + i], df["ymin"][fi + i], df["xmax"][fi + i], df["ymax"][fi + i], 1.0])
        classes[i] = 1.0  # df['class'][fi+1]

    return rects, classes


# save_csv flag only False if used in tracking.py!
def predict_images(
    model,
    image_path,
    output_path,
    output_csv_path,
    delta=0.3,
    theta=0.5,
    save_csv=True,
    return_csv=False,
    input_mode="Test",
):
    """Predict detection on image
    Args:
        model: detecting model object
        image_path (str): path to image
        output_path (str): output folder to write detected images to
        output_csv_path (str): output folder to write csv of detections to
        delta (float, optional): detection threshold. Defaults to 0.3.
        theta (float, optional): detection similarity threshold. Defaults to 0.5.
        save_csv (bool, optional): whether csv files of detection should be saved. Defaults to True.
        return_csv (bool, optional): whether tuple of all results should be returned at the end.
        input_mode (str, optional): differentiates between two ways of loading input image data/paths
    Returns:
        Tuple[np.ndarray]: tuple of np arrays (all_boxes, all_scores, all_classes, all_num_detections)
    """
    data = pd.DataFrame(columns=["filename", "width", "height", "class", "score", "xmin", "ymin", "xmax", "ymax"])
    all_boxes, all_scores, all_classes, all_num_detections = [], [], [], []

    if input_mode == "Test":
        image_loader = sorted(glob.glob(image_path))
    elif input_mode == "Train" or input_mode == "Val":
        df = pd.read_csv(image_path)
        image_loader = df["filename"].tolist()
        image_loader = list(dict.fromkeys(image_loader))
    else:
        image_loader = []
        logger.error("Wrong input mode!")
    for img in tqdm(image_loader):
        image_np, orig_w, orig_h = image_load_encode(img)

        # Each box represents a part of the image where a particular object was detected.
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.

        # Actual detection.
        pred_array = inference_detector(model, image_np)
        pred_output = pred_array  # for later visualization part
        pred_array = pred_array[0]  # List of ndarrays where each array is of a certain class, we have only one class
        # pred_array now contains all bboxes of class spine with each box of shape (0, 5)
        # len(pred_array) = 80 means 80 bboxes provided
        # pred_array[i] = [x_min, y_min, x_max, y_max, score]
        pred_boxes = np.asarray([[p[0], p[1], p[2], p[3]] for p in pred_array])
        pred_labels = np.zeros(len(pred_array))
        pred_scores = np.asarray([p[4] for p in pred_array])

        # find out where scores are greater than at threshold and change everything according to that
        thresh_indices = np.where(pred_scores >= delta)[0]
        pred_boxes = pred_boxes[thresh_indices]
        pred_scores = pred_scores[thresh_indices]
        pred_labels = pred_labels[thresh_indices]

        pred_boxes, pred_scores = postprocess(pred_boxes, pred_scores, theta=theta)

        if len(pred_scores) > 0:  # if img has any detections, recreate pred_output from postprocessed results for vis.
            pred_output = np.concatenate((pred_boxes, np.array([pred_scores]).T), axis=1)
            pred_output = [pred_output]  # turn np.array to list[np.array]

        # print("Predboxes (detached): ", pred_boxes)
        # print("Predscores (detached): ", pred_scores)
        # print("Pred_Output: ", pred_output)

        if return_csv:
            all_boxes.append(pred_boxes)
            all_classes.append(pred_labels)
            all_scores.append(pred_scores)
            all_num_detections.append(len(pred_scores))

        # Visualization of the results of a detection, but only if output_path is provided, can be empty
        if output_path is not None:
            # output_path = "output/prediction/MODEL/images_mmdet"
            orig_name = img.split("/")[-1].split("\\")[-1]
            img_output_path = os.path.join(output_path, orig_name)
            model.show_result(img, pred_output, score_thr=0.5, font_size=4, out_file=img_output_path)

        # always saving data to dataframe
        if save_csv:
            _ = write_to_df(
                data, img, orig_w, orig_h, output_csv_path, "spine", pred_boxes, pred_scores, disable_thresh=True
            )

        logger.info("Finished detection of image " + img + ".")

    if return_csv:
        return all_boxes, all_scores, all_classes, all_num_detections


if __name__ == "__main__":

    # logging.basicConfig(level=logging.INFO)
    start = time.time()
    args = parse_args(mode="predict")

    # if it doesn't make sense, print warning
    if args.use_csv is not None and not args.save_images:
        logger.warning(
            "As you are using csv files, not saving any detections will result in doing nothing. "
            "So images are saved."
        )
        args.save_images = True

    # save_images true/false, output None/path wo images/csvs
    model_name = args.model.split("/")[-1] if args.model.split("/")[-1] != "" else args.model.split("/")[-2]
    if args.output is None:
        args.output = os.path.join("output/prediction/", model_name, args.param_config)
    output_path = os.path.join(args.output, "images_mmdet")

    # create folder for prediction csvs if not already done
    if not args.use_csv:
        csv_path = os.path.join(args.output, "csvs_mmdet")
    else:
        csv_path = args.use_csv
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.save_images and not os.path.exists(output_path):
        os.makedirs(output_path)

    print(output_path, model_name, csv_path)
    # Path to the actual model that is used for the object detection.

    # Decide whether to predict the bboxes or to load from csv
    if not args.use_csv:
        model = load_model(args.model_type, args.use_aug, args.model_epoch, args.param_config)
    else:
        logger.info("Loading detections from csv file ...")
        df = pd.read_csv(args.model)
    after_loading_model = time.time()

    # Make prediction
    nr_imgs = len(list(glob.glob(args.input)))
    logger.info("Starting predictions ...")
    if not args.use_csv:
        predict_images(model, args.input, output_path, csv_path, args.delta, args.theta)
    else:
        changed_df = False
        for img in glob.glob(args.input):
            image_np, orig_w, orig_h = image_load_encode(img)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # At first change the folder in front of the filenames to the folder the images are inside
            if not changed_df:
                df["filename"] = [df["filename"][i].split("/")[-1] for i in range(len(df))]
                folder = img.replace(img.split("/")[-1], "")
                df["filename"] = folder + df["filename"]
                changed_df = True
            img_df = df[df.filename == img & df.score >= args.delta]

            # Read boxes, classes and scores
            rects, classes = df_to_data(img_df)

            # Visualization of the results of a detection.
            image_np = draw_boxes_predict(image_np, rects)
            orig_name = os.path.abspath(img).split("/")[-1]
            img_output_path = os.path.join(output_path, orig_name)
            cv2.imwrite(img_output_path, image_np)

    finished = time.time()
    logger.info(f"Model read in {after_loading_model - start}sec")
    logger.info(f"Predicted {nr_imgs} images in {finished - after_loading_model}sec")
