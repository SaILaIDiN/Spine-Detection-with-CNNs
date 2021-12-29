import os
import numpy as np
import shutil
import re
import pandas as pd
from train_mmdet import train_main
from train_mmdet import parser as parser_train
from references.mmdetection.tools.analysis_tools.analyze_logs import plot_curve, parse_args, load_json_logs, main


def get_training_dict(train_csv, special_term, model_type, use_aug, lr,
                      max_epochs=None, warm_up=None, steps_decay=None, momentum=None, weight_decay=None, dropout=None):
    dict_tmp = {'train_csv': train_csv,
                'special_term': special_term,
                'model_type': model_type,
                'use_aug': use_aug,
                'learning_rate': lr,
                'max_epochs': max_epochs,
                'warm_up': warm_up,
                'steps_decay': steps_decay,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'dropout': dropout,
                # all data augmentation functions need to be reset, or they will pass into the next configuration!
                'random_brightness': None,
                'random_contrast': None,
                'p_rbc': None,
                'vertical_flip': None,
                'horizontal_flip': None,
                'rotate': None
                }
    return dict_tmp


def get_data_aug_dict(random_brightness=None, random_contrast=None, p_rbc=None,
                      vertical_flip=None, horizontal_flip=None, rotate=None):
    dict_tmp = {'random_brightness': random_brightness,
                'random_contrast': random_contrast,
                'p_rbc': p_rbc,
                'vertical_flip': vertical_flip,
                'horizontal_flip': horizontal_flip,
                'rotate': rotate
                }
    return dict_tmp


def auto_loss_plotting(test_path, legend, model_type, mode, num_train_size):
    """ """
    args_plot = parse_args()
    argparse_plot_dict = vars(args_plot)
    dict_tmp = {'task': 'plot_curve',
                'json_logs': [f'{test_path}/None.log.json'],
                'keys': ['loss'],
                'legend': legend,
                'out': f'{test_path}/{model_type}_{mode}.png',
                'title': f'{model_type}_{mode}',
                'xmargin': 0.1 if mode == 'Train' else 0.15,
                'mode': mode,
                'backend': None,
                'ymax': None,
                'style': 'white',
                'num_iters_per_epoch': num_train_size}
    argparse_plot_dict.update(dict_tmp)
    main(args_plot)
    return


def length_of_train_set(dir_train_csv):
    """ """
    df_train = pd.read_csv(dir_train_csv)
    img_list = list(sorted(df_train["filename"].values.tolist()))
    re_pattern = re.compile(r'(.*)day(\d{1,2})stack(\d{1,2})-(\d{2})')  # get filenames only up to the stack number
    all_matches = []
    for img in img_list:
        matches = re_pattern.finditer(img)
        for match in matches:
            all_matches.append(match.group(0))
    img_list_unique = np.unique(all_matches)
    return len(img_list_unique)


args_train = parser_train.parse_args()
argparse_train_dict = vars(args_train)

# # # Hardcoded values for basic training setup
list_train_csv = [f"data/default_annotations/train_subsets/train_sub_{i+1}.csv" for i in range(0, 11)]
list_special_term = [f"_sub_{i+1}" for i in range(0, 11)]
test_content = "00_Test_Subsets"
# list_train_csv = [None]
# list_special_term = ['']
list_model_type = ["Cascade-RCNN"]
list_use_aug = ["False"]
val_max_epochs = 2
# list_learning_rate = [0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
list_learning_rate = [0.001]
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

# # # NOTE: build your training loops exactly for a specific training pattern
for train_csv, special_term in zip(list_train_csv, list_special_term):
    for model_type in list_model_type:
        for use_aug in list_use_aug:
            for lr in list_learning_rate:
                for warm_up in list_warm_up:
                    for momentum in list_momentum:
                        for weight_decay in list_weight_decay:
                            dict_tmp = get_training_dict(train_csv, special_term, model_type, use_aug, lr,
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
                            test_path = f"{model_type}_Plot_Analysis/{test_content}/{train_work_dir.split('/')[-1]}"
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
                            shutil.copy(from_path, to_path)
                            auto_loss_plotting(to_path, ['legend_train'], model_type, "Train",
                                               length_of_train_set(train_csv))
                            auto_loss_plotting(to_path, ['legend_val'], model_type, "Val",
                                               length_of_train_set(train_csv))
