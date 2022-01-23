""" This script performs the F1-Score analysis with averaging.
First, it collects the *_eval.csv files of the current part of the analysis (DA, Grid-Search, Final, ...).
Second, it computes the average and the standard deviation at every step.
Third, it stores the values in six new json files, meaning, mean and std for train, val and test.
Fourth, it plots the average for train, val and test separately by using custom_plotting.py.
"""
import pandas as pd
import os


def load_eval_csv_create_avg(eval_csvs, model_type=None, test_content=None, special_term=None):
    """ Takes paths for multiple runs of the same parameter configuration and computes mean and std. """
    df_averaging_collector = []
    for eval_csv in eval_csvs:
        df_eval = pd.read_csv(eval_csv)
        df_eval = df_eval[["epoch", "gt_version", "detection_threshold", "fscore", "precision", "recall",
                           "nr_detected", "nr_gt", "nr_gt_detected"]]
        df_averaging_collector.append(df_eval)

    # # Averaging process
    df_mean = (pd.concat(df_averaging_collector)
               .groupby(["epoch", "gt_version", "detection_threshold"])
               .agg(fscore=("fscore", "mean"), precision=("precision", "mean"), recall=("recall", "mean"),
                    nr_detected=("nr_detected", "mean"), nr_gt=("nr_gt", "mean"),
                    nr_gt_detected=("nr_gt_detected", "mean")))
    df_std = (pd.concat(df_averaging_collector)
              .groupby(["epoch", "gt_version", "detection_threshold"])
              .agg(fscore=("fscore", "std"), precision=("precision", "std"), recall=("recall", "std"),
                   nr_detected=("nr_detected", "std"), nr_gt=("nr_gt", "std"),
                   nr_gt_detected=("nr_gt_detected", "std")))
    output_main_path = f"{model_type}_Plot_Analysis/{test_content}/{eval_csvs[0].split('/')[-2]}_avg/"
    try:
        os.makedirs(output_main_path)
    except OSError as error:
        print(f"File path {output_main_path} already exists!")

    output_csv_mean = output_main_path + f"mean_{eval_csvs[0].split('/')[-1]}"
    output_csv_std = output_main_path + f"std_{eval_csvs[0].split('/')[-1]}"

    df_mean.to_csv(output_csv_mean, index=False)
    df_std.to_csv(output_csv_std, index=False)
    return df_mean, df_std


if __name__ == '__main__':
    eval_csvs = [f"../../../../results/Cascade-RCNN_aug_True/Test_high_and_low/" \
                 f"lr_0.001_warmup_None_momentum_0.0_L2_0.03_run_{i+1}/" \
                 f"Cascade-RCNN_aug_True_theta_0.4_delta_track_0.3_delta_eval_0.3_Test_eval.csv" for i in range(0, 5)]
    model_type = "Cascade_RCNN"
    test_content = "03_Test_high_and_low/evals_f1_score"
    df_mean_Cascade, df_std_Cascade = load_eval_csv_create_avg(eval_csvs, model_type, test_content)
