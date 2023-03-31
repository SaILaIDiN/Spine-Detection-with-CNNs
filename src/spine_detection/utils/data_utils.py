import random
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from scipy.spatial import distance as dist


@DATASETS.register_module(name="SpineDataset", force=True)
class SpineDataset(CustomDataset):
    """Custom Dataset class for spines."""

    CLASSES = ("spine",)

    def load_annotations(self, ann_csv_file):
        """This function loads csv annotation files instead of txt files or json files.
        The csv table is loaded into a dataframe and the set of unique filenames aka images is extracted.
        Then a list of dicts is grown with one entry per image containing all bounding boxes and labels of it.
        """

        ann_df = pd.read_csv(ann_csv_file, encoding="utf-8")
        # get all img names
        img_list = list(sorted(ann_df["filename"].values.tolist()))
        # make them unique
        img_list = np.unique(img_list)

        data_infos = []
        for i, f_name in enumerate(img_list):
            img_dict = {}
            img_dict["filename"] = f_name
            img_dict["width"] = ann_df[ann_df["filename"] == f_name]["width"].values.tolist()[0]
            img_dict["height"] = ann_df[ann_df["filename"] == f_name]["height"].values.tolist()[0]
            img_dict["ann"] = {}
            k = ann_df[ann_df["filename"] == f_name][["xmin", "ymin", "xmax", "ymax"]].values.tolist()
            img_dict["ann"]["bboxes"] = np.array(k).astype(np.float32)
            img_dict["ann"]["labels"] = np.zeros(len(k)).astype(np.int64)
            # label are zeros only because single class is "spine" and indexing of class labels start at 0!

            data_infos.append(img_dict)

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]["ann"]


def calc_metric_xy(
    centroid1: Optional[List] = None,
    centroid2: Optional[List] = None,
    rect1: Optional[List] = None,
    rect2: Optional[List] = None,
    metric: str = "iom",
) -> float:
    """calculates metric (usually Intersection over Minimum) between two centroids or two rects

    Args:
        centroid1 (Optional[List]): first centroid in (cX, cY, w, h, ...) format. Defaults to None.
        centroid2 (Optional[List]): second centroid in (cX, cY, w, h, ...) format. Defaults to None.
        rect1 (Optional[List]): first rect in (x1, y1, x2, y2) format. Defaults to None.
        rect2 (Optional[List]): second rect in (x1, y1, x2, y2) format. Defaults to None.
        metric (str): Metric to use, either iou or iom. Defaults to iom

    Raises:
        AttributeError: If neither two centroids nor two boxes are given
        NotImplementedError: metric need to be 'iom' or 'iou', other values are not implemented yet

    Returns:
        float: value of IoM
    """
    # if centroids are given -> calc box coordinates first before calculating everything
    if centroid1 is not None and centroid2 is not None:
        cX1, cY1, w1, h1 = centroid1[:4]
        cX2, cY2, w2, h2 = centroid2[:4]

        x11, x12, y11, y12 = cX1 - w1 / 2, cX1 + w1 / 2, cY1 - h1 / 2, cY1 + h1 / 2
        x21, x22, y21, y22 = cX2 - w2 / 2, cX2 + w2 / 2, cY2 - h2 / 2, cY2 + h2 / 2
    elif rect1 is not None and rect2 is not None:
        x11, y11, x12, y12 = rect1
        x21, y21, x22, y22 = rect2
    else:
        raise AttributeError("You have to provide either two centroids or two boxes but neither is given.")

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    x1, x2, y1, y2 = max(x11, x21), min(x12, x22), max(y11, y21), min(y12, y22)
    if x1 >= x2 or y1 >= y2:
        return 0

    intersection = (x2 - x1) * (y2 - y1)
    union = area1 + area2 - intersection

    if metric == "iom":
        return intersection / min(area1, area2)
    elif metric == "iou":
        return intersection / union
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented")


def calc_metric_z(centroid1: List, centroid2: List) -> float:
    """calculate IoM only for z-direction

    Args:
        centroid1 (List): First centroid of format (..., z1, z2)
        centroid2 (List): Second centroid of format (..., z1, z2)

    Returns:
        float: IoM in z-direction using start and end z-frames z1, z2
    """
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
    min_val = min(max_z1 - min_z1, max_z2 - min_z2) + 1

    if min_val == 0:
        return 0

    # gt has saved each spine with only one img -04.png
    # should be no problem any more
    return intersection / min_val


def calc_metric(centroid1: List, centroid2: List, metric: str = "iom") -> float:
    """Combine IoM in xy and in z-direction

    Args:
        centroid1 (List): First centroid (cX, cY, w, h, z1, z2)
        centroid2 (List): Second centroid same format
        metric (str): Metric to use, either iou or iom. Defaults to iom

    Returns:
        float: overall F_1-3D-score of both centroids
    """
    # how to combine both metrics
    iom = calc_metric_xy(centroid1, centroid2, metric=metric)
    z_iom = calc_metric_z(centroid1, centroid2)

    # use similar formula to fscore, but replace precision and recall with iom and z_iom
    # beta=low because z_iom should not count that much
    beta = 0.5
    if iom == 0 or z_iom == 0:
        # if iom != 0 and z_iom == 0:
        #     print(f"z-Problem: iom is {iom} while z_iom is {z_iom}")
        return 0
    final_score = (1 + beta**2) * (iom * z_iom) / (beta**2 * iom + z_iom)
    return final_score


def csv_to_boxes(df: pd.DataFrame) -> Tuple[List]:
    """This function collects and prepares the required data for tracking, by extracting from csv-files.
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


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    def ann_csv_into_txt(ann_csv):
        """Function to just have a txt file version of the csv annotations."""

        data = pd.read_csv(ann_csv, encoding="utf-8")
        with open("data/default_annotations/data.txt", "a+", encoding="utf-8") as f:
            for line in data.values:
                f.write(
                    (
                        str(line[0])
                        + "\t"
                        + str(line[1])
                        + "\t"
                        + str(line[2])
                        + "\t"
                        + str(line[3])
                        + "\t"
                        + str(line[4])
                        + "\t"
                        + str(line[5])
                        + "\t"
                        + str(line[6])
                        + "\t"
                        + str(line[7])
                        + "\n"
                    )
                )

    # ann_csv_into_txt("data/default_annotations/data.csv")

    def split_csv(ann_csv, shuffle_by_single=False, shuffle_by_group=False):
        """Split given csv with annotations into two csv files for either train or val."""

        data_df = pd.read_csv(ann_csv, encoding="utf-8")
        if shuffle_by_group:
            groups = [df for _, df in data_df.groupby("filename")]
            random.shuffle(groups)
            data_df = pd.concat(groups).reset_index(drop=True)
        if shuffle_by_single:
            data_df = data_df.sample(frac=1)  # for random shuffle, or just use shuffle parameter below
            # in this case some images may appear in train and val where the set of bboxes is split
            # so that some images are incompletely learned -> keep val mAP very low!
        train, val = train_test_split(data_df, shuffle=False, test_size=0.25)
        train.to_csv("data/default_annotations/data_train.csv", encoding="utf-8", index=False)
        val.to_csv("data/default_annotations/data_val.csv", encoding="utf-8", index=False)

    split_csv("data/default_annotations/data.csv", shuffle_by_group=True)

    def load_annotations(ann_csv_file):
        """This is a directly callable version of the SpineDataset method. For prototyping."""

        ann_df = pd.read_csv(ann_csv_file, encoding="utf-8")
        # get all img names
        img_list = list(sorted(ann_df["filename"].values.tolist()))
        # make them unique
        img_list = np.unique(img_list)

        data_infos = []
        for i, f_name in enumerate(img_list):
            img_dict = {}
            img_dict["filename"] = f_name
            img_dict["width"] = ann_df[ann_df["filename"] == f_name]["width"].values.tolist()[0]
            img_dict["height"] = ann_df[ann_df["filename"] == f_name]["height"].values.tolist()[0]
            img_dict["ann"] = {}
            k = ann_df[ann_df["filename"] == f_name][["xmin", "ymin", "xmax", "ymax"]].values.tolist()
            img_dict["ann"]["bboxes"] = np.array(k).astype(np.float32)
            img_dict["ann"]["labels"] = np.zeros(len(k)).astype(np.int64)
            if i == 1:
                print(img_dict)
            data_infos.append(img_dict)

        return data_infos

    dataset_list = load_annotations("data/default_annotations/data.csv")
    # print(dataset_list)
