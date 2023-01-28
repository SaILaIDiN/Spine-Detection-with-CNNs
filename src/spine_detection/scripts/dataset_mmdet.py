import numpy as np
import pandas as pd
import random
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module(name="SpineDataset", force=True)
class SpineDataset(CustomDataset):
    """ Custom Dataset class for spines. """

    CLASSES = ('spine',)

    def load_annotations(self, ann_csv_file):
        """ This function loads csv annotation files instead of txt files or json files.
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


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split


    def ann_csv_into_txt(ann_csv):
        """ Function to just have a txt file version of the csv annotations. """

        data = pd.read_csv(ann_csv, encoding="utf-8")
        with open("data/default_annotations/data.txt", "a+", encoding="utf-8") as f:
            for line in data.values:
                f.write((str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\t' + str(line[3]) + '\t' + str(
                    line[4]) + '\t'
                         + str(line[5]) + '\t' + str(line[6]) + '\t' + str(line[7]) + '\n'))


    # ann_csv_into_txt("data/default_annotations/data.csv")


    def split_csv(ann_csv, shuffle_by_single=False, shuffle_by_group=False):
        """ Split given csv with annotations into two csv files for either train or val. """

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
        """ This is a directly callable version of the SpineDataset method. For prototyping. """

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
