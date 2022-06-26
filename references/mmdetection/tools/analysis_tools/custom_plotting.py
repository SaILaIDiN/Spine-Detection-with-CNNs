""" File with various functions to evaluate runs and plot them for different purposes.
    Note: This file contains specifically tailored settings that are hardcoded in the plot functions.
    In this way, it should be adjusted by the user with respect to his/her needs.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage.filters as ndif


def load_eval_csv_create_avg(eval_csvs, model_type=None, test_content=None, output_main_path=None, special_term=None):
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
    df_mean["gt_version"] = df_averaging_collector[0]["gt_version"].tolist()
    df_mean["epoch"] = df_averaging_collector[0]["epoch"].tolist()
    df_std = (pd.concat(df_averaging_collector)
              .groupby(["epoch", "gt_version", "detection_threshold"])
              .agg(fscore=("fscore", "std"), precision=("precision", "std"), recall=("recall", "std"),
                   nr_detected=("nr_detected", "std"), nr_gt=("nr_gt", "std"),
                   nr_gt_detected=("nr_gt_detected", "std")))
    df_std["gt_version"] = df_averaging_collector[0]["gt_version"].tolist()
    df_std["epoch"] = df_averaging_collector[0]["epoch"].tolist()

    if output_main_path is not None:
        output_main_path = output_main_path
    else:
        output_main_path = f"{model_type}_Plot_Analysis/{test_content}/{eval_csvs[0].split('/')[-2]}_avg/"
    try:
        os.makedirs(output_main_path)
    except OSError as error:
        print(f"File path {output_main_path} already exists!")

    output_csv_mean = output_main_path + f"mean_{eval_csvs[0].split('/')[-1]}"
    output_csv_std = output_main_path + f"std_{eval_csvs[0].split('/')[-1]}"

    df_mean.to_csv(output_csv_mean, index=False)
    df_std.to_csv(output_csv_std, index=False)
    return df_mean, df_std, output_csv_mean, output_csv_std


## Needs an update
def plot_f1_score(eval_filename, main_path, benchmark=None, model_name="our model", mode="single", det_threshold=0.5,
                  input_mode=None):
    """ Take the evaluation tracking file and plot the F1-Score over epochs
     Args:
        eval_filename (str): csv file from evaluation of tracking, no absolute path!
        main_path (str): main directory to eval_filename, can be combined into absolute path
        benchmark (list[float]): list of three float values of previous best model scores for horizontal line
        model_name (str): name of the model that is evaluated
        mode (str): "single" or "average" for the of F1-Score
        input_mode (str): "Test", "Train" or "Val"
    """
    df = pd.read_csv(os.path.join(main_path, eval_filename))
    for gt_version, spine_data in df.groupby("gt_version"):
        # gt_versions are sorted by default during the creation of the csv files [min, maj, max]
        x = spine_data["epoch"].tolist()
        y = spine_data["fscore"].tolist()
        plt.plot(x, y, color='g', label=f"{model_name}")
        if benchmark is not None:
            if gt_version == "min":
                bm = benchmark[0]
            elif gt_version == "max":
                bm = benchmark[2]
            else:
                bm = benchmark[1]
            plt.axhline(y=bm, color='b', linestyle='dashed', xmin=0, xmax=max(x), label="benchmark")
        plt.xticks(ticks=x[::2])
        plt.xlabel("Epoch")
        plt.ylabel("F1-Score")
        plt.title(f"{model_name}, {mode}-F1-Score, GT_{gt_version}, {input_mode}", fontsize=15)
        plt.legend()
        plt.savefig(os.path.join(main_path,
                                 f"{mode}-F1-Score_GT_{gt_version}_det_threshold_{det_threshold}_{input_mode}.png"))
        plt.clf()
    return

## Needs an update
def plot_f1_score_overlay(eval_filename_p, main_path, benchmark=None, model_name="our model", mode="single",
                          det_thresholds=None, gt_vers=None, input_mode=None):
    """ This function makes an overlay of multiple evaluations of the same model but different detection threshold """

    for det_threshold in det_thresholds:
        df = pd.read_csv(os.path.join(main_path, eval_filename_p + f"det_threshold_{det_threshold}_{input_mode}_eval.csv"))
        for gt_version, spine_data in df.groupby("gt_version"):
            if gt_version != gt_vers:
                continue
            x = spine_data["epoch"].tolist()
            x = [i*844 for i in x]
            y = spine_data["fscore"].tolist()
            plt.plot(x, y, label=f"det_thr={det_threshold}")
            plt.xticks(ticks=x[::5])
            plt.xlabel("Iter")
            plt.ylabel("F1-Score")
            plt.title(f"{model_name}, {mode}-F1-Score, GT_{gt_version}, {input_mode}", fontsize=15)
    if benchmark is not None:
        if gt_vers == "min":
            bm = benchmark[0]
        elif gt_vers == "max":
            bm = benchmark[2]
        else:
            bm = benchmark[1]
        plt.axhline(y=bm, color='b', linestyle='dashed', label="benchmark")
    plt.legend()
    plt.savefig(os.path.join(main_path, f"{mode}-F1-Score_GT_{gt_vers}_all_det_thresholds_{input_mode}.png"))
    plt.clf()

    return

## Needs an update
def plot_f1_score_comparison(list_main_path, list_param_config, list_eval_filename_part_1, list_input_mode,
                             model_name="our model", mode="single", gt_vers=None, legend=None, benchmark=None,
                             list_eval_filename_std_part_1=None):
    """ This function makes an overlay of multiple evaluations of the same network but different model parameters.
        Works if only one parameter is tweaked at a time during function call (if you generate list_param_config).
        Specific parameter configurations with more than two different parameters can be written manually.
    """

    counter = 0  # for legend
    for version in gt_vers:
        for input_mode in list_input_mode:
            for main_path in list_main_path:
                for param_config in list_param_config:
                    for i, eval_filename_part_1 in enumerate(list_eval_filename_part_1):
                        try:
                            if eval_filename_part_1.endswith('.csv'):
                                total_path = os.path.join(
                                    main_path,
                                    os.path.join(param_config, eval_filename_part_1))
                            else:
                                total_path = os.path.join(
                                    main_path,
                                    os.path.join(param_config, eval_filename_part_1 + f"{input_mode}_eval.csv"))
                            print("Total_Path", total_path)
                            df = pd.read_csv(total_path)
                            if list_eval_filename_std_part_1 is not None:
                                assert len(list_eval_filename_part_1) == len(list_eval_filename_std_part_1)
                                total_path_std = os.path.join(
                                    main_path, os.path.join(
                                        param_config, list_eval_filename_std_part_1[i]))
                                df_std = pd.read_csv(total_path_std)
                                print(df_std.iloc[0])
                        except:
                            print("Wrong path!")
                            continue
                        for gt_version, spine_data in df.groupby("gt_version"):
                            if gt_version != version:
                                continue
                            x = spine_data["epoch"].tolist()
                            x = [i * 844 for i in x]
                            # x = np.asarray([0] + x)
                            x = np.asarray(x)
                            y = spine_data["fscore"].tolist()
                            y = np.asarray(y)
                            # y = np.asarray([None] + y)
                            if list_eval_filename_std_part_1 is not None:
                                std = df_std.loc[df_std["gt_version"] == version]["fscore"].tolist()
                                std = np.asarray(std)
                            # plt.xticks(ticks=x[::tick_steps])
                            plt.xticks(fontsize=14)
                            plt.yticks(fontsize=14)
                            plt.xlabel("Iter", fontsize=15)
                            plt.ylabel("F1-Score", fontsize=15)
                            ax = plt.gca()
                            ax.set_ylim(bottom=0.5, ymax=0.9)
                            # ax.set_xlim(xmin=0)
                            ax.margins(x=0.15, y=0.1, tight=None)
                            ax.tick_params(axis="x", direction="out", length=4, labelcolor="black", width=1)
                            ax.tick_params(axis="y", direction="out", length=4, labelcolor="black", width=1)
                            plt.locator_params(axis='y', nbins=5)
                            plt.locator_params(axis='x', nbins=6)
                            plt.title(f"{model_name}, {input_mode}, {mode}-F1-Score", fontsize=15)
                            plt.rcParams.update({"font.size": 14})
                            plt.plot(x, y, label=f"{legend[counter]}", linewidth=2.0)
                            if list_eval_filename_std_part_1 is not None:
                                plt.fill_between(x, y-std, y+std, alpha=0.3)
                        counter += 1
                if benchmark is not None:
                    if gt_vers == "min":
                        bm = benchmark[0]
                    elif gt_vers == "max":
                        bm = benchmark[2]
                    else:
                        bm = benchmark[1]
                    plt.axhline(y=bm, color='b', linestyle='dashed', label="benchmark")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(main_path, f"{mode}-F1-Score_GT_{version}_{input_mode}.png"))
                plt.clf()
    return


def plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name="our model", mode="single",
                                   gt_vers=None, legend=None, benchmark=None, list_std_full_path=None, output=None,
                                   smooth=False, smoothing_size=None, ylim=None, plot_name_postfix=""):
    """ This function is a simplified version of plot_f1_score_comparison. Where each csv file path is given in one
        piece.
    """

    counter = 0  # for legend
    # # actual nested loop together with the indentations below
    # for version in gt_vers:
    #     for input_mode in list_input_mode:
    #         for i, eval_filename in enumerate(list_eval_full_path):

    for i, (version, input_mode, eval_filename) in enumerate(zip(gt_vers, list_input_mode, list_eval_full_path)):
        try:
            if eval_filename.endswith('.csv'):
                total_path = eval_filename
            else:
                total_path = eval_filename + f"{input_mode}_eval.csv"
            print("Total_Path", total_path)
            df = pd.read_csv(total_path)
            if list_std_full_path is not None:
                assert len(list_eval_full_path) == len(list_std_full_path)
                df_std = pd.read_csv(list_std_full_path[i])
                print(df_std.iloc[0])
        except:
            print("Wrong path!")
            continue
        for gt_version, spine_data in df.groupby("gt_version"):
            if gt_version != version:
                continue
            x = spine_data["epoch"].tolist()
            x = [i * 844 for i in x]
            # x = np.asarray([0] + x)
            x = np.asarray(x)
            y = spine_data["fscore"].tolist()
            y = np.asarray(y)
            if smooth:
                y = ndif.uniform_filter1d(y, size=smoothing_size) if smoothing_size else \
                    ndif.uniform_filter1d(y, size=5)
            # y = np.asarray([None] + y)
            if list_std_full_path is not None:
                std = df_std.loc[df_std["gt_version"] == version]["fscore"].tolist()
                std = np.asarray(std)
                if smooth:
                    std = ndif.uniform_filter1d(std, size=smoothing_size) if smoothing_size else \
                        ndif.uniform_filter1d(std, size=5)
            # plt.xticks(ticks=x[::tick_steps])
            plt.xticks(fontsize=19)
            plt.yticks(fontsize=19)
            plt.xlabel("Iter", fontsize=20)
            plt.ylabel("F1-Score", fontsize=20)
            ax = plt.gca()
            if ylim is not None:
                ax.set_ylim(bottom=ylim[0], ymax=ylim[1])
            else:
                ax.set_ylim(bottom=0.0, ymax=1.0)
            # ax.set_xlim(xmin=0)
            ax.margins(x=0.15, y=0.1, tight=None)
            ax.tick_params(axis="x", direction="out", length=4, labelcolor="black", width=1)
            ax.tick_params(axis="y", direction="out", length=4, labelcolor="black", width=1)
            plt.locator_params(axis='y', nbins=5)
            plt.locator_params(axis='x', nbins=6)
            #plt.title(f"{model_name}, {input_mode}, {mode}-F1-Score", fontsize=15)
            # plt.rcParams.update({"font.size": 5})  # This changes the font size inside the legend box!!!
        if legend is not None:
            plt.plot(x, y, label=f"{legend[counter]}", linewidth=2.0)
        else:
            plt.plot(x, y, linewidth=2.0)
        if list_std_full_path is not None:
            plt.fill_between(x, y-std, y+std, alpha=0.3)
        counter += 1
        if benchmark is not None:
            if gt_vers == "min":
                bm = benchmark[0]
            elif gt_vers == "max":
                bm = benchmark[2]
            else:
                bm = benchmark[1]
            plt.axhline(y=bm, color='b', linestyle='dashed', label="benchmark")
    if legend is not None:
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=2, mode="expand", borderaxespad=0., fontsize="x-large")
    plt.tight_layout()
            # actual indentation is here!
            # plt.savefig(os.path.join(output, f"{mode}-F1-Score_GT_{version}_{input_mode}_{plot_name_postfix}.png"))
    plt.savefig(os.path.join(output, f"{mode}-F1-Score_GT_{plot_name_postfix}.png"))
    plt.clf()
    return


if __name__ == "__main__":
    eval_file = "Cascade-RCNN_aug_False_eval.csv"
    main_path = "Cascade-RCNN_Plot_Analysis/10_Test_TRICK_1/lr_0.005_warmup_None_momentum_0.9_no_DA"
    plot_f1_score(eval_file, main_path, benchmark=[0.8, 0.8, 0.8], det_threshold=0.5)
