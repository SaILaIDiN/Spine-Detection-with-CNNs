import argparse
import glob
import os
import shutil
import sys
import re
import datetime

from collections import OrderedDict
from utils_FV import CentroidTracker as CT
import numpy as np
import pandas as pd


def calc_IoM_3D(centroid1, centroid2):
    # look how many of centroid1 and centroid2 z-axis overlap
    # using intersection/union, not intersection/minimum
    min_z1, max_z1 = centroid1[-2:]
    min_z2, max_z2 = centroid2[-2:]
    
    if max_z1 < min_z2 or max_z2 < min_z1:
        return 0
        
    # +1 has to be added because of how we count with both ends including!
    # if GT is visible in z-layers 5 - 8 (inclusive) and detection is in layer 8 - 9
    # they have one overlap (8), but 8 - 8 = 0 which is wrong!
    intersection = min(max_z1, max_z2) - max(min_z1, min_z2) + 1
    min_val = min(max_z1-min_z1, max_z2-min_z2) + 1

    if min_val == 0:
        return 0
    
    # gt has saved each spine with only one img -04.png
    # should be no problem any more
    return intersection/min_val


def calc_metric(centroid1, centroid2, metric='own'):
    # how to combine both metrics
    iom = calc_IoM(centroid1, centroid2, metric=metric)
    z_iom = calc_IoM_3D(centroid1, centroid2)

    # use similar formula to fscore, but replace precision and recall with iom and z_iom
    # beta=low because z_iom should not count that much
    beta = 0.5
    if iom == 0 or z_iom == 0:
        if iom != 0 and z_iom == 0:
            print(f"z-Problem: iom is {iom} while z_iom is {z_iom}")
        return 0
    final_score = (1 + beta**2) * (iom * z_iom)/(beta**2 * iom + z_iom)
    return final_score


def calc_IoM(centroid1, centroid2, metric='own'):
    cX1, cY1, w1, h1 = centroid1[:4]
    cX2, cY2, w2, h2 = centroid2[:4]
    area1 = w1*h1; area2 = w2*h2

    x11, x12, y11, y12 = cX1-w1/2, cX1+w1/2, cY1-h1/2, cY1+h1/2
    x21, x22, y21, y22 = cX2-w2/2, cX2+w2/2, cY2-h2/2, cY2+h2/2

    # directly throw out cases where both boxes do not overlap
    x1, x2, y1, y2 = max(x11, x21), min(x12, x22), max(y11, y21), min(y12, y22)
    if x1 >= x2 or y1 >= y2:
        # or area1 >= 100*100 or area2 >= 100*100:
        return 0

    intersection = (x2-x1)*(y2-y1)
    union = area1 + area2 - intersection

    if metric == 'iou':
        return intersection/union
    elif metric == 'own':
        return intersection/min(area1, area2)

# calculates centroids from tracked csv-file by averaging spines over all there occurences
def calc_centroids_given_tracking(tracking_filename, reg_expr_for_filename='(.*)SR052N1D1day1(.*)', det_thresh=0.5):
    df = pd.read_csv(tracking_filename)
    centroids = OrderedDict()

    if reg_expr_for_filename is not None:
        re_matching = re.compile(reg_expr_for_filename)
    # loop over all given grouped spine ids and generate average centroid
    for spine_id, spine_data in df.groupby("id"):
        # get only spine ids from test dataset
        real_data = spine_data[spine_data.apply(lambda row: re_matching.match(row['filename']) is not None, axis=1)] # "SR052N1D1day1" in row['filename']
        if len(real_data) == 0:
            continue
        x1, y1, x2, y2, score = np.average(real_data[["xmin", "ymin", "xmax", "ymax", "score"]], axis=0)
        all_z_numbers = real_data["filename"].apply(lambda row: int(row[-6:-4]))
        #if spine_id < 2:
        #    print(spine_id, all_z_numbers, np.min(all_z_numbers), np.max(all_z_numbers))
        cX, cY, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
        
        # no img saving necessary -> all centroids from one session (including all stacks)
        # are directly compared with each other
        # Do NOT count spine if score is lower than 50%!
        if score < det_thresh:
            continue
        # add start and end z of visible spine MISSING
        # TP, when > 50% in z-axis overlap (over minimum) -> 2-8, 4-10 -> overlap 4/7 > 0.5
        centroids[spine_id] = (cX, cY, w, h, np.min(all_z_numbers), np.max(all_z_numbers))
    
    return centroids


# Get current path to set default folders
currentPath = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model performance compared to groundtruth labels',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional arguments
    # Mandatory
    parser.add_argument('-gtf', '--gtfolder', dest='gtFolder', default='',
                        help='either list of folders for each GT you want to look at or one folder '
                             'and a list of gt_file, comma separated')
    parser.add_argument('-det', '--detfolder', dest='detFolder', default='',
                        help='folder containing your detected tracking file')
    # Optional
    parser.add_argument('-t', '--threshold', dest='iouThreshold', type=float, default=0.3, metavar='',
                        help='IOU threshold. Default 0.5')
    parser.add_argument('-m', dest='metric', default='own', metavar='',
                        help='used metric. Options are \'own\' or \'iou\'')
    parser.add_argument('-tr', '--tracking', default='',
                        help='path of used tracking file')
    parser.add_argument('-dt', '--detection-threshold', dest='det_threshold', default=0.5, metavar='',
                        help='detection threshold for real detection')
    parser.add_argument('-gt', dest='gt_file', default='output/tracking/GT/data_tracking.csv',
                        help='given a list of gtFolders, name of gt_file is enough, '
                             'otherwise a list of gt_files must be given, comma separated')
    parser.add_argument('-sp', '--savepath', dest='savePath', metavar='',
                        help='folder where the plots are saved')
    parser.add_argument('-sn', '--savename', dest='saveName', default='',
                        help='name of results file')
    parser.add_argument('-ow', '--overwrite', action='store_true',
                        help='whether to overwrite the results of the previous iteration or just append it')
    args = parser.parse_args()

    iouThreshold = args.iouThreshold

    # Arguments validation
    errors = []
    # Groundtruth folder
    all_gt_folder = args.gtFolder.split(',')
    all_gt_files = args.gt_file.split(',')
    final_gt_paths = []
    
    if len(all_gt_folder) == 0:
        all_gt_folder = ['']
    if len(all_gt_files) == 0:
        all_gt_files = ['']
    if len(all_gt_folder) == 1 and len(all_gt_files) == 1:
        final_gt_paths.append(os.path.join(all_gt_folder[0], all_gt_files[0]))
    elif len(all_gt_folder) > 1 and len(all_gt_files) == 1:
        for i in range(len(all_gt_folder)):
            final_gt_paths.append(os.path.join(all_gt_folder[i], all_gt_files[0]))
    elif len(all_gt_folder) == 1 and len(all_gt_files) > 1:
        for i in range(len(all_gt_files)):
            final_gt_paths.append(os.path.join(all_gt_folder[0], all_gt_files[i]))
    elif len(all_gt_files) > 1 and len(all_gt_files) == len(all_gt_folder):  # includes len(all_gt_folder) > 1
        for i in range(len(all_gt_files)):
            final_gt_paths.append(os.path.join(all_gt_folder[i], all_gt_files[i]))
    else:
        raise ValueError(f"The given combination of GT Folders {args.gtFolder} and Gt files {args.gt_file} "
                         f"doesn't work. Either both lists have the same length or "
                         f"one has arbitrary length while the other has length 1")
    
    nr_gts = len(final_gt_paths)
    for i in range(nr_gts):
        if not os.path.exists(final_gt_paths[i]):
            raise ValueError(f"GT Folder {final_gt_paths[i]} doesn't exist.")
    if not os.path.exists(args.detFolder):
        raise ValueError(f"Det Folder {args.detFolder} doesn't exist.")
    else:
        detFolder = args.detFolder
    if args.tracking == '':
        raise ValueError(f"It is necessary to provide a tracking file for evaluation.")
    # Validate savePath
    # Create directory to save results
    savePath = args.savePath
    if savePath is None:
        savePath = 'results'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    real_det_thresh = float(args.det_threshold)
    total_spines = []
    total_both_spines = []
    print("  Attribute  |    GT    | Tracking | Overlap ")
    print("------------------------------------------")
    for j in range(nr_gts):
        centroids1 = calc_centroids_given_tracking(final_gt_paths[j])
        centroids2 = calc_centroids_given_tracking(os.path.join(detFolder, args.tracking))

        nr_gt = len(centroids1)
        nr_det = len(centroids2)
        thresh = 0.5
        # combine both boxes
        total_spines1 = len(centroids1)
        total_spines2 = len(centroids2)

        # both_spines contains all spines and their IoM, with keys of centroids2
        both_spines = OrderedDict()
        # boxes1 are GT, boxes2 are like Predictions
        # -> compare each box2 vs all boxes of boxes1
        for key in centroids2.keys():
            # all stacks are combined therefore no comparing to same stack necessary!
            # VORAUSSETZUNG: NUR EIN SPINE IN DIESER POSITION IN 3D!
            # Andernfalls muss noch die z-Achse berucksichtigt werden!
            all_dist = [(key, other_key, calc_metric(centroids2[key], centroids1[other_key], args.metric))
                        for other_key in centroids1.keys()]
            all_dist.sort(key=lambda x: x[2], reverse=True)

            # correct centroid with highest IoM
            if len(all_dist) == 0:
                continue
            best_key, best_other_key, best_metric = all_dist[0]
            if best_metric >= thresh:
                both_spines[best_key] = (best_other_key, best_metric)
                del centroids1[best_other_key]
        print(f"{'# spines':^13s}|{nr_gt:^10d}|{nr_det:^10d}|{len(both_spines):^10d}")

        total_spines.append(total_spines1)
        total_both_spines.append(len(both_spines))

    precision = np.array(total_both_spines)/total_spines2
    recall = np.array(total_both_spines)/total_spines
    fscore = list(precision*recall*2/(precision+recall))
    if args.saveName != '':
        filename = os.path.join(savePath, args.saveName+'.csv')
    else:
        filename = os.path.join(savePath, detFolder.split('/')[-1]+'.csv')
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame()
    new_df = pd.DataFrame({
        'nr_detected': total_spines2,
        'nr_gt': total_spines,
        'nr_gt_detected': total_both_spines,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
        'detection_threshold': real_det_thresh,
        'timestamp': str(datetime.datetime.now())
    })
    for i in range(len(precision)):
        print(f"{' Precision '+str(i+1):<13s}|          |{precision[i]:^10f}|")
    for i in range(len(recall)):
        print(f"{' Recall '+str(i+1):<13s}|          |{recall[i]:^10f}|")
    for i in range(len(fscore)):
        print(f"{' F-Score '+str(i+1):<13s}|          |{fscore[i]:^10f}|")
    
    # sort columns
    new_df = new_df[['timestamp', 'detection_threshold', 'fscore', 'precision', 'recall',
                     'nr_detected', 'nr_gt', 'nr_gt_detected']]
    if args.overwrite:
        new_df.to_csv(filename, index=False)
    else:
        together = df.append(new_df, sort=False)
        together.to_csv(filename, index=False)
