""" This file imports the parser of both tracking and evaluation and updates its arguments by automatically
 generated dictionaries so that we can loop through multiple tracking and evaluation runs without the need of a command
 line interface.
"""

from tracking_mmdet import tracking_main
from tracking_mmdet import parser as parser_tracking
from evaluate_tracking_mmdet import evaluate_tracking_main
from evaluate_tracking_mmdet import parser as parser_eval_tracking


def get_tracking_dict(model_type, use_aug, epoch, use_offsets, param_config, input_mode):
    dict_tmp = {"model": model_type + '_aug_' + use_aug,
                "images": "data/raw/test_data/SR052N1D1day1stack*.png",
                #"images": "data/default_annotations/train.csv",
                #"images": "data/default_annotations/valid.csv",
                "input_mode": input_mode,
                "use_offsets": use_offsets,
                "output": None,  # reset output directory after every run
                "model_type": model_type,
                "use_aug": use_aug,
                "model_epoch": epoch,
                "param_config": param_config}
    return dict_tmp


def get_eval_tracking_dict(model_type, use_aug, epoch, param_config, det_threshold, input_mode, show_faults):
    dict_tmp = {"detFolder": "output/tracking/" + model_type + '_aug_' + use_aug,
                "gtFolder": '',
                # "gt_file": "output/tracking/GT/data_tracking_gt_min.csv,"
                #            "output/tracking/GT/data_tracking_gt_maj.csv,"
                #            "output/tracking/GT/data_tracking_gt_max.csv",
                "gt_file": "output/tracking/GT/data_tracking_min_wo_offset.csv,"
                           "output/tracking/GT/data_tracking_maj_wo_offset.csv,"
                           "output/tracking/GT/data_tracking_max_wo_offset.csv",
                #"gt_file": "output/tracking/GT/data_tracking_GT_train.csv",
                #"gt_file": "output/tracking/GT/data_tracking_GT_val.csv",
                "tracking": "AUTO",
                "saveName": "AUTO",
                "model_type": model_type,
                "use_aug": use_aug,
                "model_epoch": epoch,
                "param_config": param_config,
                "det_threshold": det_threshold,
                "input_mode": input_mode,
                "show_faults": show_faults}
    return dict_tmp


args_tracking = parser_tracking.parse_args()
argparse_tracking_dict = vars(args_tracking)

list_model_type = ["Cascade-RCNN"]
list_use_aug = ["False"]
list_epochs = ["epoch_" + str(x) for x in range(16, 17)]
use_offsets = "False"
input_mode = "Test"  # "Train", "Val", "Test"
show_faults = "True"
# Hardcoded values for parameter configuration string 'param_config'
list_learning_rate = ['0.005']
list_weight_decay = ['0.0003']
# list_learning_rate = ['0.001', '0.0001', '1e-05', '1e-06', '1e-07']  # has to be of type str because mmdetection
# translates long float numbers into 'Xe-0Y' format when config file is loaded, starts at '1e-05'
list_warm_up = [None]
list_momentum = ['0.9']
list_det_threshold = [0.5]
#list_learning_rate = ['0.01', '0.001']
#list_learning_rate = ['5e-05', '7.5e-05']
#list_learning_rate = ['0.001', '0.0001', '1e-05', '1e-06', '1e-07']  # has to be of type str because mmdetection
# translates long float numbers into 'Xe-0Y' format when config file is loaded, starts at '1e-05'
#list_det_threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]  # has to start at least with 0.5 because the tracking
# already tracks only over 0.5 confidence

args_eval_tracking = parser_eval_tracking.parse_args()
argparse_eval_tracking_dict = vars(args_eval_tracking)

for model_type in list_model_type:
    for use_aug in list_use_aug:
        # build up the relevant loops for the 'param_config' string before you proceed with epochs
        for lr in list_learning_rate:
            for warm_up in list_warm_up:
                for momentum in list_momentum:
                    for weight_decay in list_weight_decay:
                        param_config = 'lr_' + lr + '_warmup_' + str(warm_up) + '_momentum_' + momentum + \
                                       '_L2_' + weight_decay
                        for epoch in list_epochs:
                            dict_tmp = get_tracking_dict(model_type, use_aug, epoch, use_offsets, param_config, input_mode)
                            argparse_tracking_dict.update(dict_tmp)
                            tracking_main(args_tracking)
                            # try:
                            #     tracking_main(args_tracking)
                            # except:
                            #     print("Some file or path is not existent!")
                            for det_threshold in list_det_threshold:
                                dict_tmp = get_eval_tracking_dict(model_type, use_aug, epoch, param_config, det_threshold,
                                                                  input_mode, show_faults)
                                argparse_eval_tracking_dict.update(dict_tmp)
                                evaluate_tracking_main(args_eval_tracking)
                                # try:
                                #     evaluate_tracking_main(args_eval_tracking)
                                # except:
                                #     print("Some file or path is not existent!")
