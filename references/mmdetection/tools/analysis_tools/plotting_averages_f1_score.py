""" This script performs the F1-Score analysis with averaging.
First, it collects the *_eval.csv files of the current part of the analysis (DA, Grid-Search, Final, ...).
Second, it computes the average and the standard deviation at every step.
Third, it stores the values in six new json files, meaning, mean and std for train, val and test.
Fourth, it plots the average for train, val and test separately by using custom_plotting.py.
"""
import pandas as pd
import os
from custom_plotting import plot_f1_score_comparison, load_eval_csv_create_avg


def prep_Cascade_or_VFNet_with_SGD_csvs(test_content, list_train_csv, list_special_term, list_da_modes, list_model_type,
                                        list_learning_rate, list_weight_decay, list_warm_up, list_momentum, csv_name):
    """ No tweaking of parameters allowed. Collect paths of multiple runs of same parameter configuration. """
    for da_mode in list_da_modes:
        for model_type in list_model_type:
            csv_path_collector = []
            for train_csv, special_term in zip(list_train_csv, list_special_term):
                for lr in list_learning_rate:
                    for warm_up in list_warm_up:
                        for momentum in list_momentum:
                            for weight_decay in list_weight_decay:
                                run_name = 'lr_' + str(lr) + '_warmup_' + str(warm_up) + '_momentum_' + str(momentum) +\
                                           '_L2_' + str(weight_decay) + str(special_term)
                                test_path = \
                                    f"{model_type}_Plot_Analysis/{test_content}/{run_name}"
                                csv_path = test_path + "/" + csv_name  # later, expand this part for all combinations
                csv_path_collector.append(csv_path)
            dm, ds, output_csv_mean, output_csv_std = load_eval_csv_create_avg(csv_path_collector,
                                                                               test_content=test_content,
                                                                               model_type=model_type)
    return output_csv_mean, output_csv_std


def prep_Def_DETR_with_SGD_csvs(test_content, list_train_csv, list_special_term, list_da_modes, list_model_type,
                                list_learning_rate, list_weight_decay, list_warm_up, list_dropout, list_momentum,
                                csv_name):
    """ No tweaking of parameters allowed. Collect paths of multiple runs of same parameter configuration. """
    for da_mode in list_da_modes:
        for model_type in list_model_type:
            csv_path_collector = []
            for train_csv, special_term in zip(list_train_csv, list_special_term):
                for lr in list_learning_rate:
                    for warm_up in list_warm_up:
                        for val_dropout in list_dropout:
                            for momentum in list_momentum:
                                for weight_decay in list_weight_decay:
                                    run_name = 'lr_' + str(lr) + '_warmup_' + str(warm_up) + \
                                               '_dropout_' + str(val_dropout) + '_momentum_' + str(momentum) +\
                                               '_L2_' + str(weight_decay) + str(special_term)
                                    test_path = \
                                        f"{model_type}_Plot_Analysis/{test_content}/{run_name}"
                                    csv_path = test_path + "/" + csv_name  # later, expand this part for all comb.
                csv_path_collector.append(csv_path)
            dm, ds, output_csv_mean, output_csv_std = load_eval_csv_create_avg(csv_path_collector,
                                                                               test_content=test_content,
                                                                               model_type=model_type)
    return output_csv_mean, output_csv_std


def csv_name_generator():
    """ All three params """
    pass


if __name__ == '__main__':
    # # # try out load_eval_csv_create_avg()
    eval_csvs = [f"../../../../results/Cascade-RCNN_aug_True/Test_high_and_low/" \
                 f"lr_0.001_warmup_None_momentum_0.0_L2_0.03_run_{i+1}/" \
                 f"Cascade-RCNN_aug_True_theta_0.4_delta_track_0.3_delta_eval_0.3_Test_eval.csv" for i in range(0, 5)]
    model_type = "Cascade_RCNN"
    test_content = "03_Test_high_and_low/evals_f1_score"
    df_mean_Cascade, df_std_Cascade, _, _ = load_eval_csv_create_avg(eval_csvs, model_type, test_content)

    # # # try out plot_f1_score_comparison() for Cascade-RCNN
    model_type = "Cascade_RCNN"
    test_content = "03_Test_L2/evals_f1_score"
    output_csv_mean, output_csv_std = prep_Cascade_or_VFNet_with_SGD_csvs(
        test_content=test_content,
        list_train_csv=[f"../../../../data/default_annotations/train.csv" for i in range(0, 1)],
        list_special_term=[f"_run_{i+1}" for i in range(0, 1)],
        list_da_modes=["mixed_DA"],
        list_model_type=["Cascade_RCNN"],
        list_learning_rate=[0.005],
        list_weight_decay=[0.0003],
        list_warm_up=[None],
        list_momentum=[0.9],
        csv_name="Cascade-RCNN_aug_True_theta_0.4_delta_track_0.2_delta_eval_0.3_Test_eval.csv")

    output_csv_mean2, output_csv_std2 = prep_Cascade_or_VFNet_with_SGD_csvs(
        test_content=test_content,
        list_train_csv=[f"../../../../data/default_annotations/train.csv" for i in range(0, 1)],
        list_special_term=[f"_run_{i+1}" for i in range(0, 1)],
        list_da_modes=["mixed_DA"],
        list_model_type=["Cascade_RCNN"],
        # list_learning_rate=[0.000001],
        list_learning_rate=[0.005],
        list_weight_decay=[0.03],
        list_warm_up=[None],
        list_momentum=[0.9],
        csv_name="Cascade-RCNN_aug_True_theta_0.4_delta_track_0.05_delta_eval_0.3_Test_eval.csv")

    # # # try out plot_f1_score_comparison() for VFNet
    # model_type = "VFNet"
    # test_content = "03_Test_high_and_low/evals_f1_score"
    # output_csv_mean, output_csv_std = prep_Cascade_or_VFNet_with_SGD_csvs(
    #     test_content=test_content,
    #     list_train_csv=[f"../../../../data/default_annotations/train.csv" for i in range(0, 4)],
    #     list_special_term=[f"_run_{i+1}" for i in range(0, 4)],
    #     list_da_modes=["spatial_DA"],
    #     list_model_type=["VFNet"],
    #     list_learning_rate=[0.01],
    #     list_weight_decay=[0.0003],
    #     list_warm_up=[None],
    #     list_momentum=[0.0],
    #     csv_name="VFNet_aug_True_theta_0.4_delta_track_0.3_delta_eval_0.3_Test_eval.csv")
    #
    # output_csv_mean2, output_csv_std2 = prep_Cascade_or_VFNet_with_SGD_csvs(
    #     test_content=test_content,
    #     list_train_csv=[f"../../../../data/default_annotations/train.csv" for i in range(0, 4)],
    #     list_special_term=[f"_run_{i+1}" for i in range(0, 4)],
    #     list_da_modes=["spatial_DA"],
    #     list_model_type=["VFNet"],
    #     list_learning_rate=[0.00001],
    #     list_weight_decay=[0.03],
    #     list_warm_up=[None],
    #     list_momentum=[0.9],
    #     csv_name="VFNet_aug_True_theta_0.4_delta_track_0.3_delta_eval_0.3_Test_eval.csv")


    # # # try out plot_f1_score_comparison() for Def-DETR
    # model_type = "Def_DETR"
    # test_content = "03_Test_high_and_low/evals_f1_score"
    # output_csv_mean, output_csv_std = prep_Def_DETR_with_SGD_csvs(
    #     test_content=test_content,
    #     list_train_csv=[f"../../../../data/default_annotations/train.csv" for i in range(0, 5)],
    #     list_special_term=[f"_run_{i+1}" for i in range(0, 5)],
    #     list_da_modes=["mixed_DA"],
    #     list_model_type=["Def_DETR"],
    #     list_learning_rate=[0.0001],
    #     list_weight_decay=[0.000003],
    #     list_warm_up=[None],
    #     list_dropout=[0.1],
    #     list_momentum=[0.6],
    #     csv_name="Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Test_eval.csv")
    #
    # output_csv_mean2, output_csv_std2 = prep_Def_DETR_with_SGD_csvs(
    #     test_content=test_content,
    #     list_train_csv=[f"../../../../data/default_annotations/train.csv" for i in range(0, 5)],
    #     list_special_term=[f"_run_{i+1}" for i in range(0, 5)],
    #     list_da_modes=["mixed_DA"],
    #     list_model_type=["Def_DETR"],
    #     list_learning_rate=[0.00001],
    #     list_weight_decay=[0.03],
    #     list_warm_up=[None],
    #     list_dropout=[0.1],
    #     list_momentum=[0.9],
    #     csv_name="Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Test_eval.csv")


    benchmark = [0.792, 0.785, 0.765]  # for Faster-RCNN
    list_main_path_single = [f"{model_type}_Plot_Analysis/" + test_content]
    list_param_config = [output_csv_mean.split('/')[-2], output_csv_mean2.split('/')[-2]]
    list_eval_filename_part_1 = [output_csv_mean.split('/')[-1]]
    list_input_mode = ["Test"]
    model_name = model_type
    mode = "single"
    gt_version = ["maj"]
    legend = ["L2 0.0003", "L2 0.03"]
    list_eval_filename_std_part_1 = None  # [output_csv_std.split('/')[-1]]  # name is same, because same tracking params

    plot_f1_score_comparison(list_main_path_single, list_param_config, list_eval_filename_part_1, list_input_mode,
                             model_name, mode, gt_version, legend, benchmark, list_eval_filename_std_part_1)
