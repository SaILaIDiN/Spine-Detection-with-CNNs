import argparse
import os
import glob
import cv2
import numpy as np
import pandas as pd
import scipy.io
import predict_mmdet
from pathlib import Path

from utils_FV import CentroidTracker as CT
from collections import OrderedDict
from typing import Tuple, List

parser = argparse.ArgumentParser(description='Track spines in the whole stack',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-T', '--tif', required=False,
                    help='Path to input tif stack, if this and image-flag are set, images are priorized')
parser.add_argument('-i', '--images', required=False,
                    help='Path to input images')
parser.add_argument('-t', '--threshold',
                    help='Threshold for detection', default=0.5, type=float)
parser.add_argument('-a', '--appeared',
                    help='appeared counter', default=0, type=int)
parser.add_argument('-d', '--disappeared',
                    help='disappeared counter', default=0, type=int)
parser.add_argument('-th', '--theta',
                    help='Threshold for theta (detection similarity threshold)', default=0.5, type=float)
parser.add_argument('-ta', '--tau',
                    help='Threshold for tau (tracking threshold)', default=0.3, type=float)
parser.add_argument('-m', '--model',
                    help='Path to model you want to analyze with')
parser.add_argument('-c', '--csv', required=False,
                    help='Single file or folder of csv files for previous prediction. '
                         'If this flag is set, no model prediction will be executed')

parser.add_argument('-s', '--save-images', action='store_true',
                    help='Activate this flag if images should be saved')
parser.add_argument('-o', '--output', required=False,
                    help='Path where tracking images and csv should be saved, default: output/tracking/MODEL')
parser.add_argument('-f', '--file-save',
                    help="Name of tracked data csv file", default="data_tracking")
parser.add_argument('-mc', '--metric', default='iom',
                    help='Metric which should be used for evaluating. Currently available: iom, iou. '
                         'Own metric can be implemented as lambda function which takes two arguments and returns one.')
parser.add_argument('-uo', '--use_offsets', default='False',
                    help='whether offsets should be used or not')
# For load_model()
parser.add_argument('-mt', '--model_type',
                    help='decide which model to use as config and checkpoint file. '
                         'use one of [Cascade_RCNN, GFL, VFNet, Def_DETR]')
parser.add_argument('-ua', '--use-aug', default='False',
                    help='decide to load the config file with or without data augmentation')
parser.add_argument('-me', '--model_epoch', default='epoch_1',
                    help='decide the epoch number for the model weights. use the format of the default value')
parser.add_argument('-pc', '--param_config', default='',
                    help='string that contains all parameters intentionally tweaked during optimization')
parser.add_argument('-im', '--input_mode', default='Test',
                    help='defines the proper way of loading either train, val or test data as input')


def draw_boxes(img: np.ndarray, objects: OrderedDict) -> np.ndarray:
    """ Draw boxes onto image
    Args:
        img (np.ndarray): image input to draw on
        objects (OrderedDict): Dictionary of objects of format (cX, cY, w, h, conf)
    Returns:
        np.ndarray: output image with drawn boxes
    """

    for key in objects:
        # w, h = 512, 512
        cX, cY, width, height, conf = objects[key]
        x1, x2 = int(cX - width / 2), int(cX + width / 2)
        y1, y2 = int(cY - height / 2), int(cY + height / 2)
        # correct colored rectangle
        # opencv : BGR!!!! NO RGB!!
        # linear from (0,0,255) to (255,255,0)

        # color = (255*(1-conf), 255*conf, 255*conf)
        color = (0, 255, 0)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

        # green filled rectangle for text
        color = (0, 255, 0)
        text_color = (0, 0, 0)
        img = cv2.rectangle(img, (x1, y1), (x1 + 25, y1 - 12), color, thickness=-1)

        # text
        img = cv2.putText(img, '{:02.0f}%'.format(conf * 100), (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                          text_color, 1)
    return img


def csv_to_boxes(df):
    """ This function collects and prepares the required data for tracking, by extracting from csv-files.
        These files are from previous predictions. This way, one can review older predictions multiple times.
    Args:
        df (pd.DataFrame): Dataframe of interest
    Returns:
        Tuple[List]: Tuple containing boxes, scores, classes, num detections
    """

    boxes, scores, classes = [], [], []
    for i in range(len(df)):
        if len(df.iloc[i]) == 8:
            filename, w, h, class_name, x1, y1, x2, y2 = df.iloc[i]
            score = 1.0
        else:
            filename, w, h, class_name, score, x1, y1, x2, y2 = df.iloc[i]
        scores.append(score)
        classes.append(1)  # all are spines
        boxes.append([x1, y1, x2, y2])
    boxes = [boxes]
    scores = [scores]
    classes = [classes]
    num_detections = [len(scores[0])]
    return boxes, scores, classes, num_detections


def tracking_main(args):
    # Max diff -> (minimum) diff so that two following bboxes are connected with each other
    # iom thresh -> min iom that two boxes are considered the same in the same frame!
    MAX_DIFF = args.tau
    IOM_THRESH = args.theta
    THRESH = args.threshold  # This threshold parameter is the same one used in predict.py's predict_images()
    MIN_APP = args.appeared
    MAX_DIS = args.disappeared
    METRIC = args.metric
    NUM_CLASSES = 1
    MAX_VOL = 2000

    if args.images is None:
        raise ValueError('You need to specify input images or input tif stack!')

    # save_folder: folder where tracking csv file will be saved
    # folder: name of folder which is used in csv file for generating filename-column
    model_name = args.model.split("/")[-1] if args.model.split("/")[-1] != "" else args.model.split("/")[-2]
    if args.output is None:
        args.output = os.path.join('output/tracking', model_name)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    img_output_path = os.path.join(args.output, 'images')
    csv_output_path = os.path.join(args.output, args.param_config)
    Path(csv_output_path).mkdir(parents=True, exist_ok=True)
    csv_output_path = os.path.join(csv_output_path, args.file_save + '_' + args.model_type + '_aug_' + args.use_aug
                                   + '_' + args.model_epoch + '_' + args.input_mode + '.csv')
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
        print("Wrong input mode!")

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
            raise ValueError('No csv files with valid prediction data are available.')
        csv_path = args.csv

    # get all boxes, scores and classes at the start if prediction is necessary:
    if args.csv is None:
        model = predict_mmdet.load_model(args.model_type, args.use_aug, args.model_epoch, args.param_config)
        # We currently disable storing prediction images for tracking. Replace None by img_output_path to activate
        # Other way of disabling is complicated in here and coupled to args.save_images which is always False here?!
        all_boxes, all_scores, all_classes, all_num_detections = predict_mmdet.predict_images(
            model, args.images, None, csv_output_path, threshold=THRESH, save_csv=False, return_csv=True,
            input_mode=args.input_mode)

    all_csv_paths = list(Path().rglob(args.csv))

    ct = CT(maxDisappeared=MAX_DIS, minAppeared=MIN_APP, maxDiff=MAX_DIFF,
            iomThresh=IOM_THRESH, maxVol=MAX_VOL, metric=METRIC)

    # get offsets if we want to use them
    if args.use_offsets == "True":
        sr, neuron, dend, day = 52, 1, 1, 1
        arrx = scipy.io.loadmat(f'data/offsets/SR{sr}N{neuron}D{dend}offsetX.mat')[f'SR{sr}N{neuron}D{dend}offsetX']
        arry = scipy.io.loadmat(f'data/offsets/SR{sr}N{neuron}D{dend}offsetY.mat')[f'SR{sr}N{neuron}D{dend}offsetY']

        # get offset for each stack
        offsets = np.array(list(zip(arrx[:, day-1], arry[:, day-1]))).astype(int)

        # double offsets so that it can easily be added to bounding boxes
        offsets = np.concatenate((offsets, offsets), axis=1)  # Sized in multiples of 512 == weight == height

        # make offset positive by subtracting possible negative offsets (including first offset of 0)
        offsets = offsets - np.min(offsets, axis=0)

    # use given prediction for all images, if csv is available
    for i, img in enumerate(all_imgs):
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
                    new_df = new_df[new_df.apply(lambda row: os.path.splitext(orig_img)[0] in row['filename'], axis=1)]
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
            print("NO BOXES!")
            continue
        else:
            print("BOXES!")

        # print("BOXES: ", boxes, type(boxes), type(boxes[0]))
        # print("SCORES: ", scores, type(scores))
        # print("NUM-DETECTIONS: ", num_detections, type(num_detections))
        image_np, orig_w, orig_h = predict_mmdet.image_load_encode(img)
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

        rects = np.array([[boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i]]
                          for i in range(num_detections) if scores[i] >= THRESH])

        # print("RECTS: ", rects, rects.shape)
        objects = ct.update(rects)  # y1,x1,y2,x2

        # Start with non-empty lists
        boxes = []
        classes = []
        scores = []

        # DO NOT USE absolute path for images!
        total_path = os.path.join(img_output_path, img.split('/')[-1])
        for key in objects:
            orig_dict = {'filename': total_path, 'width': w, 'height': h, 'class': 'spine'}

            # Making boxes, classes, scores correct
            cX, cY, width, height, conf = objects[key]
            x1, x2 = (cX - width / 2) / w, (cX + width / 2) / w
            y1, y2 = (cY - height / 2) / h, (cY + height / 2) / h
            boxes.append([x1, y1, x2, y2])
            classes.append(1)
            scores.append(conf)

            orig_dict.update({'id': key, 'ymin': round(y1 * h, 2), 'ymax': round(y2 * h, 2), 'xmin': round(x1 * w, 2),
                              'xmax': round(x2 * w, 2), 'score': conf})

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

    # delete all double elements
    all_dicts = [dict(tup) for tup in {tuple(set(elem.items())) for elem in all_dicts}]
    df = pd.DataFrame(all_dicts,
                      columns=['id', 'filename', 'width', 'height', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
    df.sort_values(by='filename', inplace=True)
    df.to_csv(csv_output_path, index=False)

    # count real spines (does NOT correspond to max_key, but to number of keys!)
    nr_all_ind = len(df.groupby('id'))
    print(f"Nr of spines found: {nr_all_ind}")

    print('[INFO] Written predictions to ' + csv_output_path + '.')
    return


if __name__ == '__main__':
    args = parser.parse_args()
