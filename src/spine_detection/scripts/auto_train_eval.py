""" This script combines the automated training and evaluation procedure of auto_training.py
and auto_eval.py and thus allows training and evaluating a complete run before deleting the weights
for efficient memory usage.
For reduction of boilerplate code, we import the needed functions including dictionaries for the parser.
"""
import os
import torch
import random
import imgaug
import shutil
from auto_training import get_training_dict, get_data_aug_dict, auto_loss_plotting, length_of_train_set
from train_mmdet import train_main
from train_mmdet import parser as parser_train

from auto_eval import get_tracking_dict, get_eval_tracking_dict
from tracking_mmdet import tracking_main
from tracking_mmdet import parser as parser_tracking
from evaluate_tracking_mmdet import evaluate_tracking_main
from evaluate_tracking_mmdet import parser as parser_eval_tracking


args_train = parser_train.parse_args()
argparse_train_dict = vars(args_train)

args_tracking = parser_tracking.parse_args()
argparse_tracking_dict = vars(args_tracking)

args_eval_tracking = parser_eval_tracking.parse_args()
argparse_eval_tracking_dict = vars(args_eval_tracking)

# # # Hardcoded values for basic training setup
delete_weights = True  # deletes the epoch.pth files after the current run is done and the log.json is secured

# # For Data Augmentation Analysis
list_train_csv = [f"data/default_annotations/train.csv" for i in range(0, 5)]
list_special_term = [f"_run_{i+1}" for i in range(0, 5)]
# list_seed_data = random.sample(range(100), 5)  # seed creation without duplicates for data sampling
# list_seed_weights = random.sample(range(100), 5)  # seed creation without duplicates for weights

test_content_list = ["00_Test_Fluctuations/all_seeds_fixed", "00_Test_Fluctuations/toggle_data_aug_seed",
                     "00_Test_Fluctuations/toggle_data_seed", "00_Test_Fluctuations/toggle_weight_seed"]
dict_all_seeds_fixed = {"data_aug_seeds": [0, 0, 0, 0, 0], "data_sample_seeds": [0, 0, 0, 0, 0],
                        "weights_seeds": [0, 0, 0, 0, 0]}
dict_toggle_data_aug_seed = {"data_aug_seeds": [0, 5, 13, 27, 43], "data_sample_seeds": [0, 0, 0, 0, 0],
                             "weights_seeds": [0, 0, 0, 0, 0]}
dict_toggle_data_seed = {"data_aug_seeds": [0, 0, 0, 0, 0], "data_sample_seeds": [0, 5, 13, 27, 43],
                         "weights_seeds": [0, 0, 0, 0, 0]}
dict_toggle_weight_seed = {"data_aug_seeds": [0, 0, 0, 0, 0], "data_sample_seeds": [0, 0, 0, 0, 0],
                           "weights_seeds": [0, 5, 13, 27, 43]}
dict_all_seeds_random = {"data_aug_seeds": [0, 5, 13, 27, 43], "data_sample_seeds": [0, 5, 13, 27, 43],
                         "weights_seeds": [0, 5, 13, 27, 43]}
seed_dict_list = [dict_all_seeds_fixed, dict_toggle_data_aug_seed, dict_toggle_data_seed, dict_toggle_weight_seed,
                  dict_all_seeds_random]

list_model_type = ["Cascade-RCNN"]
list_use_aug = ["False"]
val_max_epochs = 1
list_learning_rate = [0.001, 0.0001]
list_weight_decay = [0.0003]
list_warm_up = [None]  # can use 'constant', 'linear', 'exp' or None
# val_steps_decay = [5, 7]  # format [step_1, step_2, ..]
val_steps_decay = None
val_dropout = 0.5
list_momentum = [0.9]

# # # Hardcoded values for data augmentation
val_vertical_flip = None  # 0.5
val_horizontal_flip = None  # 0.5
val_rotate = None  # 0.5
val_brightness_limit = None  # [0.1, 0.3]
val_contrast_limit = None  # [0.1, 0.3]
val_p_rbc = None  # 0.2

# # # Hardcoded values for basic evaluation subprocess
list_epochs = ["epoch_" + str(x) for x in range(16, 17)]  # range has to be within val_max_epochs
use_offsets = "True"
input_mode = "Test"  # "Train", "Val", "Test"
show_faults = "False"
list_sim_threshold_track = [0.5]
list_det_threshold_track = [0.2]
list_det_threshold_eval = [0.2, 0.4, 0.6]  # values only make sense, when delta_eval >= delta_track
# NOTE: if tracking file does not hold any more entries due to high det_threshold_track or the det_threshold_eval
# is filtering out all tracked entries, evaluate_tracking_mmdet.py's evaluate_tracking_main() will throw KeyError


# Develop the loop as for auto_training.py
for test_content, seed_dict in zip(test_content_list, seed_dict_list):
    for train_csv, special_term, seed_data_aug, seed_data, seed_weights in zip(list_train_csv, list_special_term,
                                                                               seed_dict["data_aug_seeds"],
                                                                               seed_dict["data_sample_seeds"],
                                                                               seed_dict["weights_seeds"]):
        for model_type in list_model_type:
            for use_aug in list_use_aug:
                for lr in list_learning_rate:
                    for warm_up in list_warm_up:
                        for momentum in list_momentum:
                            for weight_decay in list_weight_decay:
                                random.seed(seed_data_aug)
                                imgaug.random.seed(seed_data_aug)
                                torch.manual_seed(seed_weights)
                                torch.cuda.manual_seed_all(seed_weights)
                                torch.backends.cudnn.deterministic = True
                                torch.backends.cudnn.benchmark = False
                                dict_tmp = get_training_dict(train_csv, special_term, model_type, use_aug,
                                                             seed_data, None, lr,
                                                             val_max_epochs, warm_up, val_steps_decay,
                                                             dropout=val_dropout, momentum=momentum,
                                                             weight_decay=weight_decay)
                                argparse_train_dict.update(dict_tmp)
                                if use_aug == "True":
                                    dict_tmp = get_data_aug_dict(vertical_flip=val_vertical_flip,
                                                                 horizontal_flip=val_horizontal_flip, rotate=val_rotate,
                                                                 random_brightness=val_brightness_limit,
                                                                 random_contrast=val_contrast_limit, p_rbc=val_p_rbc)
                                    argparse_train_dict.update(dict_tmp)
                                train_work_dir = train_main(args_train)
                                param_config = train_work_dir.split('/')[-1]

                                test_path = f"{model_type}_Plot_Analysis/{test_content}/{param_config}"
                                from_path = train_work_dir + "/None.log.json"
                                to_path = "references/mmdetection/tools/analysis_tools/" + test_path
                                try:
                                    os.makedirs(from_path)
                                except OSError as error:
                                    print(f"File path {from_path} already exists!")
                                try:
                                    os.makedirs(to_path)
                                except OSError as error:
                                    print(f"File path {to_path} already exists!")
                                try:
                                    os.remove(to_path + "/None.log.json")
                                except OSError as error:
                                    print(f"File None.log.json not found in {to_path}!")
                                shutil.copy(from_path, to_path)

                                # # # Start evaluation subprocess of F1-Scores
                                savePath = to_path + "/evals_f1_score"
                                try:
                                    os.makedirs(savePath)
                                except OSError as error:
                                    print(f"File path {savePath} already exists!")

                                for epoch in list_epochs:
                                    for theta in list_sim_threshold_track:
                                        for delta in list_det_threshold_track:
                                            dict_tmp = get_tracking_dict(model_type, use_aug, epoch, use_offsets, theta,
                                                                         delta, param_config, input_mode)
                                            argparse_tracking_dict.update(dict_tmp)
                                            tracking_main(args_tracking)
                                            for det_threshold in list_det_threshold_eval:
                                                dict_tmp = get_eval_tracking_dict(savePath, model_type, use_aug, epoch,
                                                                                  param_config,
                                                                                  theta, delta, det_threshold,
                                                                                  input_mode, show_faults)
                                                argparse_eval_tracking_dict.update(dict_tmp)
                                                # evaluate_tracking_main(args_eval_tracking)
                                                # # Uncomment after debugging
                                                try:
                                                    evaluate_tracking_main(args_eval_tracking)
                                                except:
                                                    print("Some file or path is not existent, "
                                                          "or some tracking or eval thresholds are too high!")
                                # # # End of evaluation subprocess

                                if delete_weights:
                                    for i in range(1, val_max_epochs+1):
                                        if os.path.isfile(train_work_dir + f"/epoch_{i}.pth"):
                                            os.remove(train_work_dir + f"/epoch_{i}.pth")
                                    [os.remove(os.path.join(train_work_dir, f)) for f in os.listdir(train_work_dir)]
                                auto_loss_plotting(to_path, ['legend_train'], model_type, "Train",
                                                   length_of_train_set(train_csv))
                                auto_loss_plotting(to_path, ['legend_val'], model_type, "Val",
                                                   length_of_train_set(train_csv))
