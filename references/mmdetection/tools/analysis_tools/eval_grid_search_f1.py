# UPDATE this docstring
""" This script performs the automated evaluation of best models from grid-search.
First, we create the paths of each run. (Two-staged because of multiple evals per param-config)
Second, we store the best three validation loss values, their epoch and the run name in a dictonary.
Third, we create a sorted csv file of all the top 3 epoch over each run.
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.mplot3d import Axes3D


def load_eval_csv_filter_top_N_single(eval_csv_path_sub, mode, top_N=1, lr=None, momentum=None, weight_decay=None,
                                      dropout=None, theta=None, delta_track=None, list_delta_eval=None, model_type=None,
                                      data_aug=None, smoothing=None):
    """ Filter out the best N f1-score values in a single training configuration for all its eval parameters.
        Apply moving averages if necessary to smooth out and remove single extreme peaks.
        Then sort the best f1-scores from all eval parameters.
    """
    # First, create the full path via tracking parameters
    list_full_paths = []
    for delta_eval in list_delta_eval:
        full_path_tmp = os.path.join(eval_csv_path_sub, f"{model_type}_aug_{data_aug}_theta_{theta}_delta_track_"
                                                        f"{delta_track}_delta_eval_{delta_eval}_{mode}_eval.csv")
        if os.path.isfile(full_path_tmp):
            list_full_paths.append(full_path_tmp)
        else:
            print(f"CSV does not exist for {full_path_tmp}!")
            continue
    # Now open each csv file, load it in a dataframe and apply moving averages
    dict_top_N_collector = []
    for eval_csv_path in list_full_paths:
        df_tmp = pd.read_csv(eval_csv_path)
        df_tmp = df_tmp[["epoch", "detection_threshold", "fscore"]]
        # df_tmp['smoothed_f1'] = df_tmp['fscore']
        df_tmp['smoothed_f1'] = df_tmp['fscore'].rolling(window=smoothing).mean()
        # print(df_tmp)
        df_top_N = df_tmp.sort_values(by=['smoothed_f1'], ascending=False).iloc[:top_N]
        # print(df_top_N)
        df_top_N["run_name"] = [eval_csv_path.split('/')[-2] for i in range(0, top_N)]
        df_top_N["lr"] = lr
        df_top_N["momentum"] = momentum
        df_top_N["weight_decay"] = weight_decay
        if dropout is not None:
            df_top_N["dropout"] = dropout
        list_dict_top_N = df_top_N.reset_index().to_dict('records')
        # print(list_dict_top_N)
        [dict_top_N_collector.append(dict_top_N) for dict_top_N in list_dict_top_N]
    # print(dict_top_N_collector)
    df_total_top_N_single = pd.DataFrame(dict_top_N_collector)
    if df_total_top_N_single.empty:
        return None
    # print(df_total_top_N_single)
    df_total_top_N_single = df_total_top_N_single.sort_values(by=['smoothed_f1'], ascending=False)
    # print(df_total_top_N_single[:top_N].reset_index().to_dict('records'))

    return df_total_top_N_single[:top_N].reset_index().to_dict('records')  # Note: these are the top_N across all eval parameter settings, not from one!


def prep_Cascade_or_VFNet_with_SGD_csvs_top_N(test_content, list_train_csv, list_special_term,
                                              list_model_type, list_learning_rate, list_weight_decay,
                                              list_warm_up, list_momentum, top_N, theta, delta_track, list_delta_eval,
                                              data_aug, smoothing):
    total_list_of_top_N = []
    for lr in list_learning_rate:
        for model_type in list_model_type:
            for train_csv, special_term in zip(list_train_csv, list_special_term):
                for warm_up in list_warm_up:
                    for momentum in list_momentum:
                        for weight_decay in list_weight_decay:
                            param_config = 'lr_' + str(lr) + '_warmup_' + str(warm_up) + \
                                           '_momentum_' + str(momentum) + '_L2_' + str(weight_decay) + \
                                           str(special_term)
                            test_path = f"{model_type}_Plot_Analysis/{test_content}/{param_config}"
                            #main_path = "references/mmdetection/tools/analysis_tools/" + test_path
                            total_path = test_path + "/" + f"evals_f1_score/{model_type}_aug_{data_aug}/{param_config}"
                            top_N_single = load_eval_csv_filter_top_N_single(total_path, mode="Val", top_N=top_N,
                                                                                  lr=lr, momentum=momentum,
                                                                                  weight_decay=weight_decay,
                                                                                  theta=theta, delta_track=delta_track,
                                                                                  list_delta_eval=list_delta_eval,
                                                                                  model_type=model_type,
                                                                                  data_aug=data_aug,
                                                                                  smoothing=smoothing)
                            if top_N_single is not None:
                                [total_list_of_top_N.append(top_N_single_sub) for top_N_single_sub in top_N_single]
    df_total_top_N_all = pd.DataFrame(total_list_of_top_N)
    df_total_top_N_all = df_total_top_N_all.sort_values(by=['smoothed_f1'], ascending=False)

    return df_total_top_N_all


def prep_Def_DETR_with_SGD_csvs_top_N(test_content, list_train_csv, list_special_term,
                                              list_model_type, list_learning_rate, list_weight_decay,
                                              list_warm_up, list_momentum, list_dropout, top_N,
                                              theta, delta_track, list_delta_eval,
                                              data_aug, smoothing):
    total_list_of_top_N = []
    for lr in list_learning_rate:
        for model_type in list_model_type:
            for train_csv, special_term in zip(list_train_csv, list_special_term):
                for warm_up in list_warm_up:
                    for dropout in list_dropout:
                        for momentum in list_momentum:
                            for weight_decay in list_weight_decay:
                                param_config = 'lr_' + str(lr) + '_warmup_' + str(warm_up) + '_dropout_' + \
                                               str(dropout) + \
                                               '_momentum_' + str(momentum) + '_L2_' + str(weight_decay) + \
                                               str(special_term)
                                test_path = f"{model_type}_Plot_Analysis/{test_content}/{param_config}"
                                #main_path = "references/mmdetection/tools/analysis_tools/" + test_path
                                total_path = test_path + "/" + f"evals_f1_score/{model_type}_aug_{data_aug}/{param_config}"
                                top_N_single = load_eval_csv_filter_top_N_single(total_path, mode="Val", top_N=top_N,
                                                                                      lr=lr, momentum=momentum,
                                                                                      weight_decay=weight_decay,
                                                                                      dropout=dropout,
                                                                                      theta=theta, delta_track=delta_track,
                                                                                      list_delta_eval=list_delta_eval,
                                                                                      model_type=model_type,
                                                                                      data_aug=data_aug,
                                                                                      smoothing=smoothing)
                                if top_N_single is not None:
                                    [total_list_of_top_N.append(top_N_single_sub) for top_N_single_sub in top_N_single]
    df_total_top_N_all = pd.DataFrame(total_list_of_top_N)
    df_total_top_N_all = df_total_top_N_all.sort_values(by=['smoothed_f1'], ascending=False)

    return df_total_top_N_all


def scatter_plot_4D_cascade(df, x_key, y_key, z_key, xlabel, ylabel, zlabel, output=None):
    """ 4D plot with x, y, z axis for the three hyperparameters lr, mom and L2.
        F1-score is then the scatter point described in color.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(xlabel, fontsize=15, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=15, labelpad=10)
    # ax.set_zlabel(zlabel, fontsize=15, labelpad=19)

    ax.xaxis.set_ticklabels(["0.01", "0.001", "1e-04", "1e-05"])
    ax.set_xticks([math.log10(0.01), math.log10(0.001), math.log10(0.0001), math.log10(0.00001)])

    ax.yaxis.set_ticklabels(["0.0", "0.3", "0.6", "0.9"])
    ax.set_yticks([0.0, 0.3, 0.6, 0.9])

    ax.zaxis.set_ticklabels(["0.03", "3e-04", "3e-06"])
    ax.set_zlim(math.log10(0.00000003), math.log10(3))
    ax.set_zticks([math.log10(0.03), math.log10(0.0003), math.log10(0.000003)])

    # change fontsize
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    for tick in ax.get_zaxis().get_major_ticks():
        tick.set_pad(8.)
    Z = np.log10(df[z_key].values.astype(float)).tolist()
    X = np.log10(df[x_key].values.astype(float)).tolist()
    Y = df[y_key].values.astype(float).tolist()
    f1 = df['smoothed_f1'].values.tolist()
    # f1[-1] = 0.5
    ax.set_box_aspect((2.25, 1, 1))
    img = ax.scatter(X, Y, Z, c=f1, alpha=0.7, cmap=plt.winter(), marker='o')
    fig.colorbar(img, pad=0.20, shrink=0.55)
    # plt.tight_layout()
    if output is None:
        plt.show()
    else:
        print(f'save scatter plot to: {output}')
        plt.savefig(output)
        plt.cla()


def scatter_plot_4D_vfnet(df, x_key, y_key, z_key, xlabel, ylabel, zlabel, output=None):
    """ 4D plot with x, y, z axis for the three hyperparameters lr, mom and L2.
        F1-score is then the scatter point described in color.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(xlabel, fontsize=15, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=15, labelpad=10)
    # ax.set_zlabel(zlabel, fontsize=15, labelpad=19)

    ax.xaxis.set_ticklabels(["0.001", "1e-04", "1e-05"])
    ax.set_xticks([math.log10(0.001), math.log10(0.0001), math.log10(0.00001)])

    ax.yaxis.set_ticklabels(["0.0", "0.3", "0.6", "0.9"])
    ax.set_yticks([0.0, 0.3, 0.6, 0.9])

    ax.zaxis.set_ticklabels(["0.03", "3e-04", "3e-06"])
    ax.set_zlim(math.log10(0.00000003), math.log10(3))
    ax.set_zticks([math.log10(0.03), math.log10(0.0003), math.log10(0.000003)])

    # change fontsize
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    for tick in ax.get_zaxis().get_major_ticks():
        tick.set_pad(8.)
    Z = np.log10(df[z_key].values.astype(float)).tolist()
    X = np.log10(df[x_key].values.astype(float)).tolist()
    Y = df[y_key].values.astype(float).tolist()
    f1 = df['smoothed_f1'].values.tolist()
    # f1[-1] = 0.5
    ax.set_box_aspect((2.25, 1, 1))
    img = ax.scatter(X, Y, Z, c=f1, alpha=0.7, cmap=plt.winter(), marker='o')
    fig.colorbar(img, pad=0.20, shrink=0.55)
    # plt.tight_layout()
    if output is None:
        plt.show()
    else:
        print(f'save scatter plot to: {output}')
        plt.savefig(output)
        plt.cla()


def scatter_plot_4D_detr(df, x_key, y_key, z_key, xlabel, ylabel, zlabel, output=None):
    """ 4D plot with x, y, z axis for the three hyperparameters lr, mom and L2.
        F1-score is then the scatter point described in color.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(xlabel, fontsize=15, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=15, labelpad=10)
    # ax.set_zlabel(zlabel, fontsize=15, labelpad=19)

    ax.xaxis.set_ticklabels(["0.001", "1e-04", "1e-05", "1e-06"])
    ax.set_xticks([math.log10(0.001), math.log10(0.0001), math.log10(0.00001), math.log10(0.000001)])

    ax.yaxis.set_ticklabels(["0.0", "0.3", "0.6", "0.9"])
    ax.set_yticks([0.0, 0.3, 0.6, 0.9])

    ax.zaxis.set_ticklabels(["0.03", "3e-04", "3e-06"])
    ax.set_zlim(math.log10(0.00000003), math.log10(3))
    ax.set_zticks([math.log10(0.03), math.log10(0.0003), math.log10(0.000003)])

    # change fontsize
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    for tick in ax.get_zaxis().get_major_ticks():
        tick.set_pad(8.)
    Z = np.log10(df[z_key].values.astype(float)).tolist()
    X = np.log10(df[x_key].values.astype(float)).tolist()
    Y = df[y_key].values.astype(float).tolist()
    f1 = df['smoothed_f1'].values.tolist()
    # f1[-1] = 0.5
    ax.set_box_aspect((2.25, 1, 1))
    img = ax.scatter(X, Y, Z, c=f1, alpha=0.7, cmap=plt.winter(), marker='o')
    fig.colorbar(img, pad=0.20, shrink=0.55)
    # plt.tight_layout()
    if output is None:
        plt.show()
    else:
        print(f'save scatter plot to: {output}')
        plt.savefig(output)
        plt.cla()


if __name__ == "__main__":
    # # for Cascade
    test_content = "02_Test_Grid_Search"
    list_train_csv = [f"data/default_annotations/train.csv" for i in range(0, 1)]
    list_special_term = [f"_run_{i + 1}" for i in range(0, 1)]
    list_model_type = ["Cascade-RCNN"]
    list_learning_rate = ['0.01', '0.001', '0.0001', '1e-05']
    list_weight_decay = ['0.03', '0.0003', '3e-06']
    list_warm_up = [None]
    list_momentum = ['0.0', '0.3', '0.6', '0.9']
    top_N = 1
    list_sim_threshold_track = [0.5]
    list_det_threshold_track = [0.3]
    list_det_threshold_eval = [0.3, 0.4, 0.5, 0.6, 0.7]
    data_aug = "True"
    smoothing = 10
    df_total_top_1 = prep_Cascade_or_VFNet_with_SGD_csvs_top_N(test_content, list_train_csv, list_special_term,
                                                               list_model_type, list_learning_rate, list_weight_decay,
                                                               list_warm_up, list_momentum, top_N,
                                                               theta=list_sim_threshold_track[0],
                                                               delta_track=list_det_threshold_track[0],
                                                               list_delta_eval=list_det_threshold_eval,
                                                               data_aug=data_aug, smoothing=smoothing)
    print(df_total_top_1.to_string())
    scatter_plot_4D_cascade(df_total_top_1, "lr", "momentum", "weight_decay", "lr", "mom", "L2", output="scatter_4D_Cascade.pdf")


    # # for VFNet
    test_content = "02_Test_Grid_Search"
    list_train_csv = [f"data/default_annotations/train.csv" for i in range(0, 1)]
    list_special_term = [f"_run_{i + 1}" for i in range(0, 1)]
    list_model_type = ["VFNet"]
    list_learning_rate = ['0.001', '0.0001', '1e-05']
    list_weight_decay = ['0.03', '0.0003', '3e-06']
    list_warm_up = [None]
    list_momentum = ['0.0', '0.3', '0.6', '0.9']
    top_N = 1
    list_sim_threshold_track = [0.5]
    list_det_threshold_track = [0.3]
    list_det_threshold_eval = [0.4, 0.5, 0.6, 0.7]
    data_aug = "True"
    smoothing = 10
    df_total_top_1 = prep_Cascade_or_VFNet_with_SGD_csvs_top_N(test_content, list_train_csv, list_special_term,
                                                               list_model_type, list_learning_rate, list_weight_decay,
                                                               list_warm_up, list_momentum, top_N,
                                                               theta=list_sim_threshold_track[0],
                                                               delta_track=list_det_threshold_track[0],
                                                               list_delta_eval=list_det_threshold_eval,
                                                               data_aug=data_aug, smoothing=smoothing)
    print(df_total_top_1.to_string())
    scatter_plot_4D_vfnet(df_total_top_1, "lr", "momentum", "weight_decay", "lr", "mom", "L2", output="scatter_4D_VFNet.pdf")


    # # for Def DETR
    test_content = "02_Test_Grid_Search"
    list_train_csv = [f"data/default_annotations/train.csv" for i in range(0, 1)]
    list_special_term = [f"_run_{i + 1}" for i in range(0, 1)]
    list_model_type = ["Def_DETR"]
    list_learning_rate = ['0.001', '0.0001', '1e-05', '1e-06']
    list_weight_decay = ['0.03', '0.0003', '3e-06']
    list_warm_up = [None]
    list_momentum = ['0.0', '0.3', '0.6', '0.9']
    list_dropout = ['0.1']
    top_N = 1
    list_sim_threshold_track = [0.5]
    list_det_threshold_track = [0.3]
    list_det_threshold_eval = [0.3, 0.4, 0.5, 0.6]
    data_aug = "True"
    smoothing = 10
    df_total_top_1 = prep_Def_DETR_with_SGD_csvs_top_N(test_content, list_train_csv, list_special_term,
                                                               list_model_type, list_learning_rate, list_weight_decay,
                                                               list_warm_up, list_momentum, list_dropout, top_N,
                                                               theta=list_sim_threshold_track[0],
                                                               delta_track=list_det_threshold_track[0],
                                                               list_delta_eval=list_det_threshold_eval,
                                                               data_aug=data_aug, smoothing=smoothing)
    print(df_total_top_1.to_string())
    scatter_plot_4D_detr(df_total_top_1, "lr", "momentum", "weight_decay", "lr", "mom", "L2", output="scatter_4D_DETR.pdf")
