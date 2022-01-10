""" This script performs the loss analysis with averaging.
First, it collects the None.log.json files of the current part of the analysis (DA, Final, ...).
Second, it computes the average and the standard deviation at every step.
Third, it stores the values in four new json files, meaning, mean and std for train and val.
Fourth, it plots the average for train and val separately by using analyze_logs.py.
NOTE: If you do not want to plot the std you have to hardcode the auto_loss_plotting() lines within a function
      by setting list_std_jsons_train and list_std_jsons_val to None.
NOTE: If only want to compute the comparison plots between DA-Modes without averaging, reduce the list_train_csv and
      list_special_term in the function call to one entry each in addition to the action in the previous NOTE.
NOTE: If you want to average a single config over multiple runs, for example the final model, only change list_da_modes
      to the used mode for the function call.
"""
# NOTE: The searching logic for the correct json locations is only given for the main optimizer of each model.
# NOTE: The general code structure suggests general usage for any step in model optimization, where averaging is needed

from analyze_logs import parse_args, main, load_json_logs_create_avg
import pandas as pd
import re
import numpy as np
import os


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


def avg_loss_plotting(plot_path, mean_jsons, std_jsons, legend, model_type, mode, num_train_size, yrange=None):
    """ """
    args_plot = parse_args()
    argparse_plot_dict = vars(args_plot)
    dict_tmp = {'task': 'plot_curve',
                'json_logs': mean_jsons,
                'std_jsons': std_jsons,
                'keys': ['loss'],
                'legend': legend,
                'out': f'{plot_path}/{model_type}_{mode}.png',
                'title': f'{model_type}_{mode}',
                'xmargin': 0.1 if mode == 'Train' else 0.15,
                'mode': mode,
                'backend': None,
                'yrange': yrange,
                'style': 'white',
                'num_iters_per_epoch': num_train_size}
    argparse_plot_dict.update(dict_tmp)
    main(args_plot)
    return


def prep_Cascade_or_VFNet_with_SGD_logs(test_content, list_train_csv, list_special_term, list_da_modes, list_model_type,
                                        list_learning_rate, list_weight_decay, list_warm_up, list_momentum):
    for lr in list_learning_rate:
        for da_mode in list_da_modes:
            for model_type in list_model_type:
                log_path_collector = []
                for train_csv, special_term in zip(list_train_csv, list_special_term):
                    for warm_up in list_warm_up:
                        for momentum in list_momentum:
                            for weight_decay in list_weight_decay:
                                run_name = 'lr_' + str(lr) + '_warmup_' + str(warm_up) + '_momentum_' + str(momentum) +\
                                           '_L2_' + str(weight_decay) + str(special_term)
                                test_path = \
                                    f"{model_type}_Plot_Analysis/{test_content.split('/')[0]}/{da_mode}/{run_name}"
                                log_path = test_path + "/None.log.json"
                    log_path_collector.append(log_path)
            load_json_logs_create_avg(log_path_collector, mode="Train", special_term="_"+da_mode+"_Train")
            load_json_logs_create_avg(log_path_collector, mode="Val", special_term="_"+da_mode+"_Val")
        plot_path = f"{model_type}_Plot_Analysis/Plots/LR_{lr}"
        if os.path.isdir(plot_path) is False:
            os.makedirs(plot_path)

        list_mean_jsons_train = [f"None_Mean_{da_mode}_Train.log.json" for da_mode in list_da_modes]
        list_std_jsons_train = [f"None_Std_{da_mode}_Train.log.json" for da_mode in list_da_modes]
        avg_loss_plotting(plot_path, list_mean_jsons_train, list_std_jsons_train, list_da_modes, model_type,
                          "Train", length_of_train_set(train_csv))
        list_mean_jsons_val = [f"None_Mean_{da_mode}_Val.log.json" for da_mode in list_da_modes]
        list_std_jsons_val = [f"None_Std_{da_mode}_Val.log.json" for da_mode in list_da_modes]
        avg_loss_plotting(plot_path, list_mean_jsons_val, list_std_jsons_val, list_da_modes, model_type,
                          "Val", length_of_train_set(train_csv))


def prep_Def_DETR_with_ADAM_logs(test_content, list_train_csv, list_special_term, list_da_modes, list_model_type,
                                 list_learning_rate, list_weight_decay, list_warm_up, list_dropout):
    for lr in list_learning_rate:
        for da_mode in list_da_modes:
            for model_type in list_model_type:
                log_path_collector = []
                for train_csv, special_term in zip(list_train_csv, list_special_term):
                    for warm_up in list_warm_up:
                        for val_dropout in list_dropout:
                            for weight_decay in list_weight_decay:
                                run_name = 'lr_' + str(lr) + '_warmup_' + str(warm_up) + '_dropout_' + str(val_dropout) +\
                                           '_L2_' + str(weight_decay) + str(special_term)
                                test_path = \
                                    f"{model_type}_Plot_Analysis/{test_content.split('/')[0]}/{da_mode}/{run_name}"
                                log_path = test_path + "/None.log.json"
                    log_path_collector.append(log_path)
            load_json_logs_create_avg(log_path_collector, mode="Train", special_term="_"+da_mode+"_Train")
            load_json_logs_create_avg(log_path_collector, mode="Val", special_term="_"+da_mode+"_Val")
        plot_path = f"{model_type}_Plot_Analysis/Plots/LR_{lr}"
        if os.path.isdir(plot_path) is False:
            os.makedirs(plot_path)

        list_mean_jsons_train = [f"None_Mean_{da_mode}_Train.log.json" for da_mode in list_da_modes]
        list_std_jsons_train = [f"None_Std_{da_mode}_Train.log.json" for da_mode in list_da_modes]
        avg_loss_plotting(plot_path, list_mean_jsons_train, list_std_jsons_train, list_da_modes, model_type,
                          "Train", length_of_train_set(train_csv))
        list_mean_jsons_val = [f"None_Mean_{da_mode}_Val.log.json" for da_mode in list_da_modes]
        list_std_jsons_val = [f"None_Std_{da_mode}_Val.log.json" for da_mode in list_da_modes]
        avg_loss_plotting(plot_path, list_mean_jsons_val, list_std_jsons_val, list_da_modes, model_type,
                          "Val", length_of_train_set(train_csv))


if __name__ == "__main__":
    prep_Cascade_or_VFNet_with_SGD_logs(test_content="TRY_AVG",
                                        list_train_csv=[f"../../../../data/default_annotations/train.csv" for i in range(0, 3)],
                                        list_special_term=[f"_run_{i+1}" for i in range(0, 3)],
                                        list_da_modes=["no_DA", "spatial_DA", "pixel_DA", "mixed_DA"],
                                        list_model_type=["Cascade_RCNN"],
                                        list_learning_rate=[0.001],
                                        list_weight_decay=[0.0003],
                                        list_warm_up=[None],
                                        list_momentum=[0.9])
