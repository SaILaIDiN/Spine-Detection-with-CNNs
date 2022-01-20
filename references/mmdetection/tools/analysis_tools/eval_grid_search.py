""" This script performs the automated evaluation of best models from grid-search.
First, we create the paths of each run.
Second, we store the best three validation loss values, their epoch and the run name in a dictonary.
Third, we create a sorted csv file of all the top 3 epoch over each run.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt


def load_json_log_filter_top_N(json_log, mode=None, top_N=1, lr=None, momentum=None, weight_decay=None):
    """ Similar way to read and manipulate json file entries such as load_json_logs().
        This function filters non-relevant rows of the json file and return the top three entries, with loss, and epoch.

        NOTE: this function is only processing a single log.json file.
        NOTE: this functionality is not provided for the metric 'mAP' only for 'loss'.
    """
    if mode == "Train":
        list_of_modes = ["train"]
    elif mode == "Val":
        list_of_modes = ["val"]
    else:
        list_of_modes = ["train", "val"]
    for mode_tmp in list_of_modes:

        with open(json_log, 'r') as log_file:
            log_dicts_tmp = []
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                # skip lines with 'mAP' field (val set)
                if 'mAP' in log:
                    continue
                # skip lines of the wrong mode
                if log["mode"] != mode_tmp:
                    continue
                log_dicts_tmp.append(log)  # list holds individual dicts from json file
            df_tmp = pd.DataFrame(log_dicts_tmp)  # makes this list into a dataframe
            # print(df_tmp)
            df_tmp = df_tmp[["mode", "epoch", "iter", "loss"]]
            # print(df_tmp)
            # Sorting and filtering step
            df_top_N = df_tmp.sort_values(by=['loss'], ascending=True).iloc[:top_N]
            df_top_N["run_name"] = [json_log.split('/')[-2] for i in range(0, top_N)]
            df_top_N["lr"] = lr
            df_top_N["momentum"] = momentum
            df_top_N["weight_decay"] = weight_decay
            list_dict_top_N = df_top_N.reset_index().to_dict('records')

    return list_dict_top_N


def prep_Cascade_or_VFNet_with_SGD_logs_top_N(test_content, list_train_csv, list_special_term, list_da_modes,
                                              list_model_type, list_learning_rate, list_weight_decay,
                                              list_warm_up, list_momentum, top_N):
    total_list_of_top_N = []
    for lr in list_learning_rate:
        for da_mode in list_da_modes:
            for model_type in list_model_type:
                for train_csv, special_term in zip(list_train_csv, list_special_term):
                    for warm_up in list_warm_up:
                        for momentum in list_momentum:
                            for weight_decay in list_weight_decay:
                                run_name = 'lr_' + str(lr) + '_warmup_' + str(warm_up) + '_momentum_' + str(momentum) +\
                                           '_L2_' + str(weight_decay) + str(special_term)
                                test_path = \
                                    f"{model_type}_Plot_Analysis/{test_content.split('/')[0]}/{da_mode}/{run_name}"
                                log_path = test_path + "/None.log.json"
                                list_of_top_N = load_json_log_filter_top_N(log_path, mode="Val", top_N=top_N,
                                                                           lr=lr, momentum=momentum,
                                                                           weight_decay=weight_decay)
                                [total_list_of_top_N.append(list_of_top_N_sub) for list_of_top_N_sub in list_of_top_N]
    df_total_top_N = pd.DataFrame(total_list_of_top_N)
    df_total_top_N = df_total_top_N.sort_values(by=['loss'], ascending=True)

    return df_total_top_N


def scatter_plot_loss_by_single_param(df, key_param, xlabel, ylabel, title=None, output=None, custom_dict=None):
    """ """
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # ax = plt.gca()
    # ax.set_xlim(xmin=0, xmax=2300)
    # ax.set_ylim(bottom=0, ymax=3.25)
    if custom_dict is not None:
        # custom order needed, because of str formatting of small decimals to Xe-YZ style
        df = df.sort_values(by=[key_param], ascending=True, key=lambda x: x.map(custom_dict))
    X = df[key_param].values.tolist()
    print(X)
    Y = df["loss"].values.tolist()
    print(Y)
    plt.scatter(X, Y, linewidths=0.1, alpha=0.7, s=150)
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=2, mode="expand", borderaxespad=0., fontsize="x-large")
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        print(f'save scatter plot to: {output}')
        plt.savefig(output)
        plt.cla()


if __name__ == "__main__":

    # # # Test load_json_log_filter_top_N()
    # run_name = "lr_1e-06_warmup_None_momentum_0.6_L2_0.03_run_1"
    # test_path = "Cascade_RCNN_Plot_Analysis/02_Test_Grid_Search/mixed_DA/" + run_name
    # log_path = test_path + "/None.log.json"
    #
    # example_val_loss_top_N = load_json_log_filter_top_N(log_path, mode="Val", top_N=5)
    # print(example_val_loss_top_N)

    # # Test prep_Cascade_or_VFNet_with_SGD_logs_top_N()
    # for Cascade
    # test_content = "02_Test_Grid_Search/some_folder"
    # list_train_csv = [f"data/default_annotations/train.csv" for i in range(0, 1)]
    # list_special_term = [f"_run_{i + 1}" for i in range(0, 1)]
    # list_da_modes = ["mixed_DA"]
    # list_model_type = ["Cascade_RCNN"]
    # list_learning_rate = ['0.01', '0.001', '0.0001', '1e-05', '1e-06']
    # list_weight_decay = ['0.03', '0.0003', '3e-06']
    # list_warm_up = [None]
    # list_momentum = ['0.0', '0.3', '0.6', '0.9']

    # for VFNet
    test_content = "02_Test_Grid_Search/some_folder"
    list_train_csv = [f"data/default_annotations/train.csv" for i in range(0, 1)]
    list_special_term = [f"_run_{i + 1}" for i in range(0, 1)]
    list_da_modes = ["spatial_DA"]
    list_model_type = ["VFNet"]
    list_learning_rate = ['0.01', '0.001', '0.0001', '1e-05']
    list_weight_decay = ['0.03', '0.0003', '3e-06']
    list_warm_up = [None]
    list_momentum = ['0.0', '0.3', '0.6', '0.9']

    df_total_top_1 = prep_Cascade_or_VFNet_with_SGD_logs_top_N(
        test_content, list_train_csv, list_special_term, list_da_modes, list_model_type, list_learning_rate,
        list_weight_decay, list_warm_up, list_momentum, top_N=2)
    print(df_total_top_1.to_string())

    # # Test scatter_plot_loss_by_single_param()
    scatter_plot_loss_by_single_param(df_total_top_1, "momentum", xlabel="momentum", ylabel="val loss",
                                      output="scatter_momentum_loss",
                                      custom_dict={'0.0': 0, '0.3': 1, '0.6': 2, '0.9': 3})
    scatter_plot_loss_by_single_param(df_total_top_1, "lr", xlabel="learning rate", ylabel="val loss",
                                      output="scatter_lr_loss",
                                      custom_dict={'0.01': 0, '0.001': 1, '0.0001': 2, '1e-05': 3, '1e-06': 4})
    scatter_plot_loss_by_single_param(df_total_top_1, "weight_decay", xlabel="weight decay", ylabel="val loss",
                                      output="scatter_weight_decay_loss",
                                      custom_dict={'0.03': 0, '0.0003': 1, '3e-06': 2})
