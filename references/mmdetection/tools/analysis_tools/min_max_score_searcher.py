""" This script parses through a given csv-file and extracts the minimum and maximum value.
    Currently used to get the ranges of the standard deviation.
"""
import pandas as pd
import os


def search_min_max(path_main, list_csvs):
    """ """
    min_max_collector = []
    for csv_name in list_csvs:
        csv_path = os.path.join(path_main, csv_name)
        df_tmp = pd.read_csv(csv_path)
        min = df_tmp["fscore"][20:].min()
        max = df_tmp["fscore"][20:].max()
        min_max_collector.append((round(min, 4), round(max, 4)))
    return min_max_collector


if __name__ == "__main__":
    path_main_cascade = "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations"
    list_csvs_cascade = [
        "all_seeds_fixedstd_Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
        "toggle_data_aug_seedstd_Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
        "toggle_data_seedstd_Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
        "toggle_weight_seedstd_Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
        "all_seeds_randomstd_Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv"
        ]

    path_main_vfnet = "VFNet_Plot_Analysis/00_Test_Fluctuations"
    list_csvs_vfnet = [
        "all_seeds_fixedstd_VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
        "toggle_data_aug_seedstd_VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
        "toggle_data_seedstd_VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
        "toggle_weight_seedstd_VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
        "all_seeds_randomstd_VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
        ]

    path_main_detr = "Def_DETR_Plot_Analysis/00_Test_Fluctuations"
    list_csvs_detr = [
        "all_seeds_fixedstd_Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
        "toggle_data_aug_seedstd_Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
        "toggle_data_seedstd_Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
        "toggle_weight_seedstd_Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
        "all_seeds_randomstd_Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
        ]

    min_max_cascade = search_min_max(path_main_cascade, list_csvs_cascade)
    min_max_vfnet = search_min_max(path_main_vfnet, list_csvs_vfnet)
    min_max_detr = search_min_max(path_main_detr, list_csvs_detr)

    print(min_max_cascade)
    print(min_max_vfnet)
    print(min_max_detr)
