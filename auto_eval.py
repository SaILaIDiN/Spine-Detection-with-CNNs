""" This file imports the parser of both tracking and evaluation and updates its arguments by automatically
 generated dictionaries so that we can loop through multiple tracking and evaluation runs without the need of a command
 line interface.
"""

from tracking_mmdet import tracking_main
from tracking_mmdet import parser as parser_tracking
from evaluate_tracking_mmdet import evaluate_tracking_main
from evaluate_tracking_mmdet import parser as parser_eval_tracking


def get_tracking_dict(model_type, use_aug, epoch, use_offsets):
    dict_tmp = {"model": model_type + '_aug_' + use_aug,
                "images": "data/raw/person1/SR052N1D1day1stack*.png",
                "use_offsets": use_offsets,
                "output": None,  # reset output directory after every run
                "model_type": model_type,
                "use_aug": use_aug,
                "model_epoch": epoch}
    return dict_tmp


def get_eval_tracking_dict(model_type, use_aug, epoch):
    dict_tmp = {"detFolder": "output/tracking/" + model_type + '_aug_' + use_aug,
                "gtFolder": '',
                "gt_file": "output/tracking/GT/data_tracking_gt_min.csv,"
                           "output/tracking/GT/data_tracking_gt_maj.csv,"
                           "output/tracking/GT/data_tracking_gt_max.csv",
                # "gt_file": "output/tracking/GT/data_tracking_min_wo_offset.csv,"
                #            "output/tracking/GT/data_tracking_maj_wo_offset.csv,"
                #            "output/tracking/GT/data_tracking_max_wo_offset.csv",
                "tracking": "AUTO",
                "saveName": "AUTO",
                "model_type": model_type,
                "use_aug": use_aug,
                "model_epoch": epoch}
    return dict_tmp


args_tracking = parser_tracking.parse_args()
argparse_tracking_dict = vars(args_tracking)

list_model_type = ["VFNet", "GFL", "Def_DETR"]
list_use_aug = ["True", "False"]
list_epochs = ["epoch_" + str(x) for x in range(1, 5)]
use_offsets = "True"

args_eval_tracking = parser_eval_tracking.parse_args()
argparse_eval_tracking_dict = vars(args_eval_tracking)

for model_type in list_model_type:
    for use_aug in list_use_aug:
        if model_type == "Def_DETR" and use_aug == "True":
            continue  # because this model has no data augmentation
        for epoch in list_epochs:
            dict_tmp = get_tracking_dict(model_type, use_aug, epoch, use_offsets)
            argparse_tracking_dict.update(dict_tmp)
            try:
                tracking_main(args_tracking)
            except:
                print("Some file or path is not existent!")
            dict_tmp = get_eval_tracking_dict(model_type, use_aug, epoch)
            argparse_eval_tracking_dict.update(dict_tmp)
            try:
                evaluate_tracking_main(args_eval_tracking)
            except:
                print("Some file or path is not existent!")