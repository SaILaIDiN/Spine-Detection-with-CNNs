import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

from mmdet.apis import init_detector, inference_detector
from mmcv import Config

import numpy as np
import os
import glob
import sys
import cv2
import argparse
import pandas as pd
import time
from utils_FV import calc_metric_xy
from typing import List, Optional, Tuple
from pathlib import Path

sys.path.append("..")

parser = argparse.ArgumentParser(description='Make prediction on images')
parser.add_argument('-m', '--model',
                    help='Model used for prediction (without frozen_inference_graph.pb!) or folder '
                         'where csv files are saved')
parser.add_argument('-t', '--delta',
                    help='Threshold for delta (detection threshold, score level)', default=0.5, type=float)
parser.add_argument('-th', '--theta',
                    help='Threshold for theta (detection similarity threshold)', default=0.5, type=float)
parser.add_argument('-C', '--use_csv', action='store_true',
                    help='activate this flag if you want to use the given csv files')
parser.add_argument('-i ', '--input',
                    help='Path to input image(s), ready for prediction. '
                         'Path can contain wildcards but must start and end with "')
parser.add_argument('-s', '--save_images', action='store_true',
                    help='Activate this flag if images should be saved')
parser.add_argument('-o', '--output', required=False,
                    help='Path where prediction images and csvs should be saved', default='output/prediction/MODEL')
# For load_model() use_aug, model_epoch
parser.add_argument('-mt', '--model_type',
                    help='decide which model to use as config and checkpoint file. '
                         'use one of [Cascade_RCNN, GFL, VFNet, Def_DETR]')
parser.add_argument('-ua', '--use_aug', default='False',
                    help='decide to load the config file with or without data augmentation')
parser.add_argument('-me', '--model_epoch', default='epoch_1',
                    help='decide the epoch number for the model weights. use the format of the default value')
parser.add_argument('-pc', '--param_config', default='',
                    help='string that contains all parameters intentionally tweaked during optimization')


def image_load_encode(img_path):
    """ Load image from path to 512x512 format
    Args:
        img_path (str): path to image file
    Returns:
        Tuple[np.ndarray, int, int]: image as np-array, its width and height
    """
    # function to read img from given path and convert to get correct 512x512 format
    # new_img = np.zeros((512, 512, 3))
    # new_img[:256, :] = image[:, :512]
    # new_img[256:, :] = image[:, 512:]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    return img.copy(), w, h


def image_decode(img=None, rect=None):
    """ Reverse image encoding potentially applied to rects as well
    Args:
        img (Optional[np.ndarray]): input (512x512) image to decode
        rect (Optional[List]): rect in (x1, y1, x2, y2) format to decode
    Raises:
        AttributeError: At least img or rect must be not None to get a result
    Returns:
        np.ndarray: Depending on the non-None inputs decoded output of either img, rect or (img, rect)
    """
    # function to decode img or detection, depending which type is provided to get original img/detection back
    # rects have x/y values between 0 and 512 and are of type xmin, ymin, xmax, ymax
    # convert img back to 1024/256
    # img = np.zeros((256, 1024, 3))
    # img[:, :512] = orig_img[:256, :]
    # img[:, 512:] = orig_img[256:, :]
    if img is None and rect is None:
        raise AttributeError(
            "At least one of img or rect need to have not None values.")
    if img is None:
        return np.array(rect).astype(int)
    if rect is None:
        return img
    else:
        return img, rect


def postprocess(boxes, scores, theta=0.5):
    """ Postprocess boxes and scores and average boxes if necessary
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
                      scores[indices])
            max_score = np.max(scores[indices])

            # score calculates as follows: max_score + weight*extra, weight = 1-max_score (to stay <= 1)
            # extra = sum(weight_i * score_i) with weight_i = score_i/(sum(scores) - max_score)
            # extra is <= 1 as well, it corresponds to the average of scores without max_score
            new_score = max_score + (1 - max_score) * (np.sum(scores[indices] ** 2) - max_score ** 2) / (
                        np.sum(scores[indices]) - max_score)
            final_boxes.append(new_box.astype(np.int64))
            final_scores.append(new_score)
        else:
            final_boxes.append(boxes[indices[0]])
            final_scores.append(scores[indices[0]])
    return np.array(final_boxes), np.array(final_scores)


def write_to_df(df, img_path, w, h, csv_path, class_label, boxes, scores, thresh=0.0, disable_thresh=False):
    """ Write detection to dataframe
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
        dict_list.append({'filename': img_path, 'width': w, 'height': h, 'class': class_label,
                          'score': scores[i], 'xmin': box[0], 'ymin': box[1], 'xmax': box[2], 'ymax': box[3]})
    if len(dict_list) != 0:
        df = df.append(dict_list)
    # be aware of windows adding \\ for folders in paths!
    csv_filepath = os.path.join(csv_path, Path(img_path).name[:-4] + '.csv')
    df.to_csv(csv_filepath, index=False)
    print("[INFO] Detections saved in csv file "+csv_filepath+".")
    return df


def draw_boxes(orig_img, boxes, scores, thresh=0.3, disable_thresh=False):
    """ Draw detection boxes onto image
    Args:
        orig_img (np.ndarray): original image to draw on
        boxes (np.ndarray): detection boxes
        scores (np.ndarray): detection confidence scores
        thresh (float, optional): min confidence necessary to count as spine. Defaults to 0.3.
        disable_thresh (bool, optional): Flag whether to use differentiation by confidence score. Defaults to False.
    Returns:
        np.ndarray:
    """
    img = image_decode(img=orig_img)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        conf = scores[i]
        x1, y1, x2, y2 = image_decode(rect=(x1, y1, x2, y2))
        if not disable_thresh and conf < thresh:
            continue

        # correct colored rectangle
        # opencv : BGR!!!! NO RGB!!
        # linear from (0,0,255) to (255,255,0)
        # color = (255*(1-conf), 255*conf, 255*conf)
        color = (0, 255, 0)
        text_color = (0, 0, 0)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)

        # green filled rectangle for text and adding border as well
        # width of rect depends on width of text
        text_width = 23 if conf < 0.995 else 30
        img = cv2.rectangle(img, (x1, y1), (x1 + text_width, y1 - 12), color, thickness=-1)
        img = cv2.rectangle(img, (x1, y1), (x1 + text_width, y1 - 12), color, thickness=1)

        # text
        img = cv2.putText(img, '{:02.0f}%'.format(conf * 100), (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                          text_color, 1)
    return img


def df_to_data(df):
    """ Converts GT dataframe to detections and their classes with confidences 1.0
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
    w, h = df['width'][fi], df['height'][fi]
    for i in range(len(df)):
        rects[i] = np.array([df['xmin'][fi + i], df['ymin'][fi + i], df['xmax'][fi + i], df['ymax'][fi + i], 1.0])
        classes[i] = 1.0  # df['class'][fi+1]

    return rects, classes


def load_model(model_type, use_aug, model_epoch, param_config):
    """ Load frozen model
        Args:
            param_config (str): contains a pregenerated string of the tweaked hyperparameters used to navigate through
                                model folders
    """

    model_folder = "tutorial_exps"
    print("[INFO] Loading model ...")
    if model_type == "Cascade-RCNN":
        if use_aug == "True":
            checkpoint_file = os.path.join(model_folder, "Cascade_RCNN_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco_SPINE_AUG.py")
        else:
            checkpoint_file = os.path.join(model_folder, "Cascade_RCNN_no_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco_SPINE.py")
    elif model_type == "GFL":
        if use_aug == "True":
            checkpoint_file = os.path.join(model_folder, "GFL_RX101_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_SPINE_AUG.py")
        else:
            checkpoint_file = os.path.join(model_folder, "GFL_RX101_no_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_SPINE.py")
    elif model_type == "VFNet":
        if use_aug == "True":
            checkpoint_file = os.path.join(model_folder, "VFNet_RX101_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_SPINE_AUG.py")
        else:
            checkpoint_file = os.path.join(model_folder, "VFNet_RX101_no_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_SPINE.py")
    elif model_type == "Def_DETR":
        if use_aug == "True":
            checkpoint_file = os.path.join(model_folder, "Def_DETR_R50_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco_SPINE_AUG.py")
        else:
            checkpoint_file = os.path.join(model_folder, "Def_DETR_R50_no_data_augmentation")
            config_file = Config.fromfile(
                "references/mmdetection/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco_SPINE.py")
    else:
        checkpoint_file = os.path.join(model_folder, "Cascade_RCNN")
        config_file = Config.fromfile(
           "references/mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco_SPINE.py")

    # construct checkpoint file name
    checkpoint_file = os.path.join(checkpoint_file, os.path.join(param_config, model_epoch + ".pth"))

    device = 'cuda:0'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    return model


# save_csv flag only False if used in tracking.py!
def predict_images(model, image_path, output_path, output_csv_path, delta=0.3, theta=0.5, save_csv=True,
                   return_csv=False, input_mode="Test"):
    """ Predict detection on image
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
    data = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
    all_boxes, all_scores, all_classes, all_num_detections = [], [], [], []

    if input_mode == "Test":
        image_loader = sorted(glob.glob(image_path))
    elif input_mode == "Train" or input_mode == "Val":
        df = pd.read_csv(image_path)
        image_loader = df["filename"].tolist()
        image_loader = list(dict.fromkeys(image_loader))
    else:
        image_loader = []
        print("Wrong input mode!")
    for img in image_loader:
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
            output_path = "output/prediction/MODEL/images_mmdet"
            orig_name = img.split('/')[-1].split('\\')[-1]
            img_output_path = os.path.join(output_path, orig_name)
            model.show_result(img, pred_output, score_thr=0.5, font_size=4, out_file=img_output_path)

        # always saving data to dataframe
        if save_csv:
            _ = write_to_df(data, img, orig_w, orig_h, output_csv_path, 'spine', pred_boxes, pred_scores,
                            disable_thresh=True)

        print('[INFO] Finished detection of image ' + img + '.')

    if return_csv:
        return all_boxes, all_scores, all_classes, all_num_detections


if __name__ == '__main__':
    start = time.time()
    args = parser.parse_args()

    # if it doesn't make sense, print warning
    if args.use_csv is not None and not args.save_images:
        print("[WARNING] As you are using csv files, not saving any detections will result in doing nothing. "
              "So images are saved.")
        args.save_images = True

    # save_images true/false, output None/path wo images/csvs
    model_name = args.model.split("/")[-1] if args.model.split("/")[-1] != "" else args.model.split("/")[-2]
    if args.output is None:
        args.output = os.path.join("output/prediction/", model_name)
    output_path = os.path.join(args.output, "images")

    # create folder for prediction csvs if not already done
    if not args.use_csv:
        csv_path = os.path.join(args.output, 'csvs_mmdet')
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
    # PATH_TO_CKPT = os.path.join('own_models2', args.model, 'faster_rcnn_model.pth')

    # Decide whether to predict the bboxes or to load from csv
    if not args.use_csv:
        model = load_model(args.model_type, args.use_aug, args.model_epoch, args.param_config)
    else:
        print("[INFO] Loading detections from csv file ...")
        df = pd.read_csv(args.model)
    after_loading_model = time.time()

    # Make prediction
    nr_imgs = len(list(glob.glob(args.input)))
    print("[INFO] Starting predictions ...")
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
                df['filename'] = [df['filename'][i].split('/')[-1] for i in range(len(df))]
                folder = img.replace(img.split('/')[-1], '')
                df['filename'] = folder + df['filename']
                changed_df = True
            img_df = df[df.filename == img & df.score >= args.delta]

            # Read boxes, classes and scores
            rects, classes = df_to_data(img_df)

            # Visualization of the results of a detection.
            image_np = draw_boxes(image_np, rects)
            orig_name = os.path.abspath(img).split('/')[-1]
            img_output_path = os.path.join(output_path, orig_name)
            cv2.imwrite(img_output_path, image_np)

    finished = time.time()
    print(f"Model read in {after_loading_model - start}sec")
    print(f"Predicted {nr_imgs} images in {finished - after_loading_model}sec")
