import argparse
import os
import glob
import cv2
import numpy as np
import pandas as pd
import time
import scipy.io
import predict

from utils_FV import CentroidTracker as CT
from collections import OrderedDict

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
    help='disappeared counter', default=1, type=int)
parser.add_argument('-th', '--theta',
    help='Threshold for theta (detection similarity threshold)', default=0.5, type=float)
parser.add_argument('-ta', '--tau',
    help='Threshold for tau (tracking threshold)', default=0.3, type=float)
parser.add_argument('-m', '--model',
    help='Path to model you want to analyze with')
# parser.add_argument('-lm', '--labelmap',
#     help='Path to labelmap', default='data/spine_label_map.pbtxt')
parser.add_argument('-c', '--csv', required=False,
    help='Single file or folder of csv files for previous prediction. '
         'If this flag is set, no model prediction will be executed')

parser.add_argument('-s', '--save_images', action='store_true',
    help='Activate this flag if images should be saved')
parser.add_argument('-o', '--output', required=False,
    help='Path where tracking images and csv should be saved, default: output/tracking/MODEL')
parser.add_argument('-f', '--file_save',
    help="Name of tracked data csv file", default="data_tracking.csv")
#parser.add_argument('-g', '--gpu',
#    help='GPU Number')
#parser.add_argument('-n', '--no_prediction',
#    help='NOT IMPLEMENTED Make prediction on images again/ y/n')
parser.add_argument('-mc', '--metric', default='iom',
    help='Metric which should be used for evaluating. Currently available: iom, iou. '
         'Own metric can be implemented as lambda function which takes two arguments and returns one.')


def draw_boxes2(img, objects):
    return np.zeros((1,1))


def draw_boxes(img, objects):
    for key in objects:
        #w, h = 512, 512
        cX, cY, width, height, conf = objects[key]
        x1, x2 = int(cX-width/2), int(cX+width/2)
        y1, y2 = int(cY-height/2), int(cY+height/2)
        # correct colored rectangle
        # opencv : BGR!!!! NO RGB!!
        # linear from (0,0,255) to (255,255,0)
        
        # color = (255*(1-conf), 255*conf, 255*conf)
        color = (0,255,0)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
        
        # green filled rectangle for text
        color=(0, 255, 0)
        text_color = (0, 0, 0)
        img = cv2.rectangle(img, (x1, y1), (x1+25, y1-12), color, thickness=-1)
    
        # text
        img = cv2.putText(img, '{:02.0f}%'.format(conf*100), (x1+2, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
    return img


def csv_to_boxes(df):
    """ This function collects and prepares the required data for tracking, by extracting from csv-files.
        These files are from previous predictions. This way, one can review older predictions multiple times.
    """
    boxes, scores, classes = [], [], []
    for i in range(len(df)):
        if len(df.iloc[i]) == 8:
            filename, w, h, class_name, x1, y1, x2, y2 = df.iloc[i]
            score = 1.0
        else:
            filename, w, h, class_name, score, x1, y1, x2, y2 = df.iloc[i]
        scores.append(score)
        classes.append(1) # all are spines
        boxes.append([x1/w, y1/h, x2/w, y2/h])
        # boxes are in y1, x1, y2, x2 format!!!
        # boxes.append([y1/h, x1/w, y2/h, x2/w])
        # boxes.append([x1, y1, x2, y2])
    boxes = [boxes]
    scores = [scores]
    classes = [classes]
    num_detections = [len(scores[0])]
    return boxes, scores, classes, num_detections


if __name__ == '__main__':
    start = time.time()
    args = parser.parse_args()
    # Max diff -> (minimum) diff so that two following bboxes are connected with each other
    # iom thresh -> min iom that two boxes are considered the same in the same frame!
    MAX_DIFF = args.tau
    IOM_THRESH = args.theta
    THRESH = args.threshold  # This threshold parameter is the same one used in predict.py's predict_images()
    MIN_APP = args.appeared
    MAX_DIS = args.disappeared
    METRIC = args.metric
    # PATH_TO_LABELS = args.labelmap
    NUM_CLASSES = 1
    MAX_VOL = 2000

    # args.save, args.output
    
    # df = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    if args.tif is None and args.images is None:
        raise ValueError('You need to specify input images or input tif stack!')

    # save_folder: folder where tracking csv file will be saved
    # folder: name of folder which is used in csv file for generating filename-column
    model_name = args.model.split("/")[-1] if args.model.split("/")[-1] != "" else args.model.split("/")[-2]
    if args.output is None:
        args.output = os.path.join('output/tracking', model_name)  # os.path.join(args.save, args.model.split('/')[-1]
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    img_output_path = os.path.join(args.output, 'images')
    csv_output_path = os.path.join(args.output, args.file_save)
    if args.save_images and not os.path.exists(img_output_path):
        os.makedirs(img_output_path)

    # to get some annotations on the first images too, make the same backwards
    all_imgs = sorted(glob.glob(args.images))

    all_dicts = []
    total_boxes = []
    total_classes = []
    total_scores = []
    nr_imgs = len(list(all_imgs))
    start_imgs = time.time()
    objects = dict()
    
    # set default values for current_stack and max_key
    # this should not be necessary as they are set in the first iteration, but not used there (if i != 0)
    # however, python return error
    # Problem is not in the first if-clause but after the for loop
    
    # if it's just a single csv file, load all data before iterating over images
    if args.csv is not None:
        all_csv_files = glob.glob(args.csv)
        if len(all_csv_files) == 0:
            raise ValueError('No csv files with valid prediction data are available.')
        csv_path = args.csv 

    # get all boxes, scores and classes at the start if prediction is necessary:
    if args.csv is None:
        model = predict.load_model(args.model)
        all_boxes, all_scores, all_classes, all_num_detections = predict.predict_images(model, args.images,
                                                                                        img_output_path,
                                                                                        csv_output_path,
                                                                                        threshold=THRESH,
                                                                                        save_csv=False, return_csv=True)

    ct = CT(maxDisappeared=MAX_DIS, minAppeared=MIN_APP, maxDiff=MAX_DIFF,
            iomThresh=IOM_THRESH, maxVol=MAX_VOL, metric=METRIC)

    # use given prediction for all images, if csv is available
    for i, img in enumerate(all_imgs):
        orig_img = img.split('/')[-1]
        if args.csv is not None:  # PART with no new prediction, instead use csv output from previous prediction
            print(1)
            if os.path.splitext(args.csv)[1] != '.csv':  # Case when args.csv is only the directory without file name
                print(2)
                csv_path = os.path.join(args.csv, os.path.splitext(orig_img)[0]+'.csv')
                try:
                    print(3)
                    new_df = pd.read_csv(csv_path)
                    boxes, scores, classes, num_detections = csv_to_boxes(new_df)
                except:
                    print(4)
                    continue
            else:  # Case when csv-file path is correct
                try:
                    print(5)
                    if len(glob.glob(args.csv)) <= 1:  # Case for single csv-file, here meaning single image
                        print(5.001)
                        new_df = pd.read_csv(args.csv)  # # # CODE Crashes here if args.csv contains a file name pattern
                        # # # but works if a single correct name is given!!! -> pd.read_csv does not handle patterns
                        # # # like glob.glob() does! -> used if condition to separate both cases

                    else:  # Case for multiple csv-files, here meaning multiple images
                        # initialize an empty DataFrame of correct columns to allow list comprehension
                        # of course, building the same DataFrame for every image is strongly redundant/inefficient
                        new_df = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'score',
                                                       'xmin', 'ymin', 'xmax', 'ymax'])
                        print("len df: ", len(new_df))
                        print(5.01)

                        df_list = [new_df.append(pd.read_csv(path_csv)) for path_csv in sorted(glob.glob(args.csv))]
                        print("df_list: ", len(df_list))
                        new_df = pd.concat(df_list)  # Dataframes cannot use append() inplace -> concat the list comp.
                        print("len df: ", len(new_df))
                        print(5.02)
                    # load only data from interesting image
                    print(5.1)
                    new_df = new_df[new_df.apply(lambda row: os.path.splitext(orig_img)[0] in row['filename'], axis=1)]
                    # axis=1 for looping through rows, to remove the '.png' extension in the filename
                    print(5.2)
                    # Dataframe must start with index 0 -> change row index
                    # new_df.index = range(len(new_df))
                    # not necessary if using iloc instead of loc!

                    boxes, scores, classes, num_detections = csv_to_boxes(new_df)
                    print(5.3)
                    print("Values: ", len(boxes), len(scores), num_detections)
                except:
                    print(6)
                    continue
        else:
            print(7)
            # just load data from saved list
            # this works as all_imgs from this file and sorted(glob.glob(args.images)) from predict sort all
            # image paths so they are perfectly aligned
            # NOTE: the output values from the prediction are of type np.ndarray
            boxes, scores, classes, num_detections = all_boxes[i], all_scores[i], all_classes[i], all_num_detections[i]

        # look if there are some boxes
        if len(boxes) == 0:
            print("NO BOXES!")
            continue
        else:
            print("BOXES!")

        print("BOXES SHAPE: ", boxes.shape, type(boxes))
        print("SCORES: ", scores, type(scores))
        print("NUM-DETECTIONS: ", num_detections, type(num_detections))

        image_np, orig_w, orig_h = predict.image_load_encode(img)
        h, w = image_np.shape[:2]

        # # # Real tracking part!
        # the if-condition with THRESH manages both origins of boxes, scores and num_detections to become equal now
        # this is because the prediction output delivers the unfiltered number of detections,
        # while the csv created in predict_images() is already filtered by the same threshold as THRESH!
        rects = np.array([[boxes[i][0]*w, boxes[i][1]*h, boxes[i][2]*w, boxes[i][3]*h, scores[i]]
                          for i in range(num_detections) if scores[i] >= THRESH])
            
        objects = ct.update(rects)  # y1,x1,y2,x2

        # Start with non-empty lists
        boxes = []
        classes = []
        scores = []

        # DO NOT USE absolute path for images!
        total_path = os.path.join(img_output_path, img.split('/')[-1])
        # real_output_path = os.path.join(save_folder, img.split('/')[-1])
        # print('keys:', objects.keys())
        for key in objects:
            orig_dict = {'filename': total_path, 'width': w, 'height': h, 'class': 'spine'}

            # Making boxes, classes, scores correct
            cX, cY, width, height, conf = objects[key]
            x1, x2 = (cX-width/2)/w, (cX+width/2)/w
            y1, y2 = (cY-height/2)/h, (cY+height/2)/h
            boxes.append([x1, y1, x2, y2])
            classes.append(1)
            scores.append(conf)

            orig_dict.update({'id': key, 'ymin': round(y1*h, 2), 'ymax': round(y2*h, 2), 'xmin': round(x1*w, 2),
                              'xmax': round(x2*w, 2), 'score': conf})

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
    end_imgs = time.time()
    
    # print('All dicts:', all_dicts)
    # delete all double elements
    all_dicts = [dict(tup) for tup in {tuple(set(elem.items())) for elem in all_dicts}]
    df = pd.DataFrame(all_dicts, columns=['id', 'filename', 'width', 'height', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
    df.sort_values(by='filename', inplace=True)
    df.to_csv(csv_output_path, index=False)
    
    # count real spines (does NOT correspond to max_key, but to number of keys!)
    nr_all_ind = len(df.groupby('id'))
    print(f"Nr of spines found: {nr_all_ind}")
    
    print('[INFO] Written predictions to '+csv_output_path+'.')
    total_end = time.time()
    
    # with open("log_tracking.txt", "w+") as f:
    #     f.write("Without writing and reading images")
    #     f.write(f"Needed {start_imgs-start}sec for starting")
    #     f.write(f"Tracking {nr_imgs} imgs in one direction in {end_imgs-start_imgs}sec -> {(end_imgs-start_imgs)/nr_imgs}sec per image")
    #     f.write(f"Tracking {nr_imgs} imgs backwards in {total_end-end_imgs}sec -> {(total_end-end_imgs)/nr_imgs}sec per image")
