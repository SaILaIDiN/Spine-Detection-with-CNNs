import argparse
import os
import re
import datetime

from collections import OrderedDict
from utils_FV import calc_metric
import numpy as np
import pandas as pd
from predict_mmdet import load_model


# calculates centroids from tracked csv-file by averaging spines over all there occurrences
def calc_centroids_given_tracking(tracking_filename: str, reg_expr_for_filename: str = '(.*)SR052N1D1day1(.*)',
                                  det_thresh: float = 0.5) -> OrderedDict:
    """ Calculate centroids for specific images only
     Args:
        tracking_filename (str): path to already tracked file
        reg_expr_for_filename (str, optional): regular expression to get only the images you want to evaluate on.
            Defaults to '(.*)SR052N1D1day1(.*)'.
        det_thresh (float, optional): detection confidence threshold. Defaults to 0.5.
    Returns:
        OrderedDict: id, rect pairs of centroids
    """
    df = pd.read_csv(tracking_filename)
    df["id"] = range(0, len(df))  # make every spine unique even if it is the same within a stack

    centroids = OrderedDict()

    if reg_expr_for_filename is not None:
        re_matching = re.compile(reg_expr_for_filename)
    # loop over all given grouped spine ids and generate average centroid
    for spine_id, spine_data in df.groupby("id"):
        # get only spine ids from test dataset
        # the next line is to filter out incomplete or wrong lines in the dataframe
        real_data = spine_data[spine_data.apply(lambda row: re_matching.match(row['filename']) is not None, axis=1)]
        # "SR052N1D1day1" in row['filename']
        if len(real_data) == 0:
            continue
        x1, y1, x2, y2, score = np.average(real_data[["xmin", "ymin", "xmax", "ymax", "score"]], axis=0)
        filename = real_data["filename"].to_numpy()[0]  # Not very elegant?
        raw_box = [x1, y1, x2, y2]
        all_z_numbers = real_data["filename"].apply(lambda row: int(row[-6:-4]))
        cX, cY, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

        # no img saving necessary -> all centroids from one session (including all stacks)
        # are directly compared with each other
        # Do NOT count spine if score is lower than 50%!
        if score < det_thresh:
            continue
        # add start and end z of visible spine MISSING
        # TP, when > 50% in z-axis overlap (over minimum) -> 2-8, 4-10 -> overlap 4/7 > 0.5
        # NOTE: Only the first four and last two outputs are relevant for metric computation of centroids
        # so for additional throughput of information one can fill arbitrary number of data in between
        centroids[spine_id] = (cX, cY, w, h, raw_box, score, filename, np.min(all_z_numbers), np.max(all_z_numbers))

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
    parser.add_argument('-ot', '--overlap-threshold', dest='overlap_threshold', type=float, default=0.5, metavar='',
                        help='IoM threshold, defines when a prediction box and a GT box overlap enough to be matched.'
                             'Impacts the number of total overlaps shown in the evaluation print statement.')
    parser.add_argument('-dt', '--detection-threshold', dest='det_threshold', default=0.5, metavar='',
                        help='Detection threshold for real detection, defines which spines will be tracked.'
                             'Impacts the number of total tracked spines shown in the evaluation print statement.'
                             'Also impacts the total number of overlaps as a secondary consequence.')
    parser.add_argument('-m', dest='metric', default='iom', metavar='',
                        help='used metric. Options are \'iom\' or \'iou\'')
    parser.add_argument('-tr', '--tracking', default='',
                        help='path of used tracking file')
    parser.add_argument('-gt', dest='gt_file', default='output/tracking/GT/data_tracking_min_wo_offset.csv',
                        help='given a list of gtFolders, name of gt_file is enough, '
                             'otherwise a list of gt_files must be given, comma separated')
    parser.add_argument('-sp', '--savepath', dest='savePath', metavar='',
                        help='folder where the plots are saved')
    parser.add_argument('-sn', '--savename', dest='saveName', default='',
                        help='name of results file')
    parser.add_argument('-ow', '--overwrite', action='store_true',
                        help='whether to overwrite the results of the previous iteration or just append it')
    # Optional, for Advanced Behavioral Analysis
    parser.add_argument('-sf', '--show-faults', dest='show_faults', default='False',
                        help='Boolean value, to select analysis of locations for false positives and missing GTs.')
    # parser.add_argument('-fbp', '--faulty-boxes-path', dest='faulty_boxes_path', default='',
    #                     help='Path where the images containing boxes of false positives and missing GTs are stored.')
    args = parser.parse_args()

    if args.show_faults == "True":
        model = load_model()

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
        thresh = args.overlap_threshold
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
            all_dist.sort(key=lambda x: x[2], reverse=True)  # sort by metric (here: IoM)

            # correct centroid with highest IoM
            if len(all_dist) == 0:
                continue
            best_key, best_other_key, best_metric = all_dist[0]
            if best_metric >= thresh:
                # PC: both_spines[prediction_key] = (best_GT_key, their_overlap)
                both_spines[best_key] = (best_other_key, best_metric)
                del centroids1[best_other_key]  # to not assign this GT box to another detection box
                # so it is first-come-first-serve

        if args.show_faults:
            # # Get GT boxes missed by predictions
            # transfer OrderedDict into Dataframe (for easier grouping by filename)
            n_GT = len(centroids1.keys())
            df_GT = pd.DataFrame(centroids1, columns=centroids1.keys()).T
            df_GT = df_GT.rename(columns={0: "cX", 1: "cY", 2: "w", 3: "h", 4: "raw_box", 5: "score", 6: "filename",
                                          7: "z_min", 8: "z_max"})

            for filename, spine_data in df_GT.groupby("filename"):

                raw_boxes = spine_data["raw_box"].to_numpy(copy=True)
                raw_boxes = np.stack(raw_boxes)
                scores = spine_data["score"].to_numpy(copy=True)

                boxes_scores = np.array(np.column_stack((raw_boxes, scores)), dtype="float32")
                # print("Boxes Scores: ", boxes_scores)  # np.array([[x1, y1, x2, y2, score], ..])
                boxes_scores = [boxes_scores]  # turn np.array to list[np.array]

                raw_img_path = "data/raw/person1"
                img_input_path = filename.split('/')[-1]
                img_input_path = os.path.join(raw_img_path, img_input_path)
                output_path = "output/tracking/BehaviorAnalysis/images_mmdet_GT"
                orig_name = img_input_path.split('/')[-1].split('\\')[-1]
                img_output_path = os.path.join(output_path, orig_name)
                model.show_result(img_input_path, boxes_scores, bbox_color="green", score_thr=0.5, font_size=3,
                                  thickness=1, out_file=img_output_path)
            # # Get false positive prediction boxes
            TP_keys = list(both_spines.keys())  # True Positives
            P_keys = list(centroids2.keys())  # All Positives
            n_FP = len(P_keys) - len(TP_keys)
            for key in TP_keys:
                del centroids2[key]
            # transfer OrderedDict into Dataframe (for easier grouping by filename)
            df_PRED = pd.DataFrame(centroids2, columns=centroids2.keys()).T
            df_PRED = df_PRED.rename(columns={0: "cX", 1: "cY", 2: "w", 3: "h", 4: "raw_box", 5: "score", 6: "filename",
                                              7: "z_min", 8: "z_max"})

            for filename, spine_data in df_PRED.groupby("filename"):

                raw_boxes = spine_data["raw_box"].to_numpy(copy=True)
                raw_boxes = np.stack(raw_boxes)  # Remove the inner lists from the outer np.array
                scores = spine_data["score"].to_numpy(copy=True)

                boxes_scores = np.array(np.column_stack((raw_boxes, scores)), dtype="float32")
                # print("Boxes Scores: ", boxes_scores)  # np.array([[x1, y1, x2, y2, score], ..])
                boxes_scores = [boxes_scores]  # turn np.array to list[np.array]

                raw_img_path = "data/raw/person1"
                img_input_path = filename.split('/')[-1]
                img_input_path = os.path.join(raw_img_path, img_input_path)
                output_path = "output/tracking/BehaviorAnalysis/images_mmdet_PRED"
                orig_name = img_input_path.split('/')[-1].split('\\')[-1]
                img_output_path = os.path.join(output_path, orig_name)
                model.show_result(img_input_path, boxes_scores, bbox_color="red", score_thr=0.5, font_size=3,
                                  thickness=1, out_file=img_output_path)

        print(f"{'# spines':^13s}|{nr_gt:^10d}|{nr_det:^10d}|{len(both_spines):^10d}")
        print("n_FP: ", n_FP)
        print("n_GT: ", n_GT)
        total_spines.append(total_spines1)
        total_both_spines.append(len(both_spines))

    precision = np.array(total_both_spines) / total_spines2
    recall = np.array(total_both_spines) / total_spines
    fscore = list(precision * recall * 2 / (precision + recall))
    if args.saveName != '':
        filename = os.path.join(savePath, args.saveName + '.csv')
    else:
        filename = os.path.join(savePath, detFolder.split('/')[-1] + '.csv')
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
        print(f"{' Precision ' + str(i + 1):<13s}|          |{precision[i]:^10f}|")
    for i in range(len(recall)):
        print(f"{' Recall ' + str(i + 1):<13s}|          |{recall[i]:^10f}|")
    for i in range(len(fscore)):
        print(f"{' F-Score ' + str(i + 1):<13s}|          |{fscore[i]:^10f}|")

    # sort columns
    new_df = new_df[['timestamp', 'detection_threshold', 'fscore', 'precision', 'recall',
                     'nr_detected', 'nr_gt', 'nr_gt_detected']]
    if args.overwrite:
        new_df.to_csv(filename, index=False)
    else:
        together = df.append(new_df, sort=False)
        together.to_csv(filename, index=False)