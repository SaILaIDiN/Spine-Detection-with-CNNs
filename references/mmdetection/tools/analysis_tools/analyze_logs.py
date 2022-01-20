import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()


def plot_curve(log_dicts, args, std_dicts=None):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    if args.mode == "TrainVal":
        assert len(legend) == (len(args.json_logs) * 2 * len(args.keys))
        # because each json file creates two log_dict files one for train one for val
    else:
        assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            if args.mode == "TrainVal":
                mode = 'val' if log_dict[1]['mode'][0] == 'val' else 'train'
                print(f'plot curve of {args.json_logs[min(i, abs(int(i-len(log_dicts)/2)))]}, '
                      f'mode is {mode}, metric is {metric}')
            else:
                print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            try:
                tmp = log_dict[epochs[0]]
            except:
                print("Sample size higher than train set size! So no train values stored.")
                continue
            if metric not in log_dict[epochs[0]]:
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}')

            if 'mAP' in metric:
                xs = np.arange(1, max(epochs) + 1)
                ys = []
                for epoch in epochs:
                    ys += log_dict[epoch][metric]
                ax = plt.gca()
                ax.set_xticks(xs)
                plt.xlabel('epoch')
                plt.ylabel('mAP')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
            else:
                xs = []
                ys = []
                if std_dicts is not None:
                    stds = []
                # num_iters_per_epoch = log_dict[epochs[0]]['iter'][-2]
                if args.num_iters_per_epoch is None:
                    num_iters_per_epoch = 844
                else:
                    num_iters_per_epoch = args.num_iters_per_epoch
                for epoch in epochs:
                    iters = log_dict[epoch]['iter']
                    # if log_dict[epoch]['mode'][-1] == 'val':
                    #     iters = iters[:-1]
                    if log_dict[epoch]['mode'][0] == 'val':
                        iters = num_iters_per_epoch
                        iters = [iters]
                    xs.append(
                        np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                    if std_dicts is not None:
                        stds.append(np.array(std_dicts[i][epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                if std_dicts is not None:
                    stds = np.concatenate(stds)
                ax = plt.gca()
                for tick in ax.xaxis.get_major_ticks():
                    tick.tick1line.set_visible(True)
                    tick.label1.set_visible(True)
                for tick in ax.yaxis.get_major_ticks():
                    tick.tick1line.set_visible(True)
                    tick.label1.set_visible(True)
                ax.get_xaxis().set_visible(True)
                ax.tick_params(axis="x", direction="out", length=4, labelcolor="black", width=1)
                ax.tick_params(axis="y", direction="out", length=4, labelcolor="black", width=1)
                plt.xlabel('Iter', fontsize=15)
                plt.ylabel('Loss', fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                # plt.rc("xtick", labelsize=15)
                # plt.rc("ytick", labelsize=15)
                plt.rcParams.update({"font.size": 14})
                plt.locator_params(axis='y', nbins=5)
                plt.locator_params(axis='x', nbins=6)
                legend_p1 = legend[i * num_metrics + j].split('_')[0]
                legend_p2 = legend[i * num_metrics + j].split('_')[1]
                plt.plot(xs, ys, label=f"{legend_p1} {legend_p2}", linewidth=2.0)
                if std_dicts is not None:
                    plt.fill_between(xs, ys-stds, ys+stds, alpha=0.3)
                if args.yrange is not None:
                    if isinstance(args.yrange, list):
                        ax.set_ylim(bottom=args.yrange[0], ymax=args.yrange[1])
                    else:
                        ax.set_ylim(bottom=0, ymax=args.yrange)
                # ax.set_xlim(xmin=0)
                if args.xmargin is not None:
                    ax.margins(x=args.xmargin, y=0.1)
            plt.legend()
        if args.title is not None:
            plt.title(f"{args.title.split('_')[0]}, {args.title.split('_')[-1]}", fontsize=15)
        plt.tight_layout()
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format. log should contain mean loss values if std_jsons is not None')
    parser_plt.add_argument(
        'std_jsons',
        type=str,
        nargs='+',
        help='path of std logs in json format',
        default=None
    )
    parser_plt.add_argument(
        '--num_iters_per_epoch',
        type=int,
        default=None,
        help='number of images per epoch aka size of current train set, for x axis of plots'
    )
    parser_plt.add_argument(
        '--mode',
        type=str,
        default=None,
        help='choose from ["Train", "Val", "TrainVal"] to search in JSON log file for correct plots'
    )
    parser_plt.add_argument(
        '--xmargin',
        type=float,
        default=None,
        help='custom change of margin of x-axis, element [0, 1]'
    )
    parser_plt.add_argument(
        '--yrange',
        type=float,
        default=None,
        help='custom change of maximum value on y-axis'
    )
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['bbox_mAP'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='white', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs, mode=None):
    """ Extracts mode-relevant entries from the json file into list of dictionaries of defaultdicts.
        Preceeding preparation of json files for plot_curve().
    """
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts_collector = []
    if mode == "Train":
        list_of_modes = ["train"]
    elif mode == "Val":
        list_of_modes = ["val"]
    else:
        list_of_modes = ["train", "val"]
    for mode_tmp in list_of_modes:
        log_dicts = [dict() for _ in json_logs]
        for json_log, log_dict in zip(json_logs, log_dicts):
            with open(json_log, 'r') as log_file:
                for line in log_file:
                    log = json.loads(line.strip())
                    # skip lines without `epoch` field
                    if 'epoch' not in log:
                        continue
                    # skip lines of the wrong mode
                    if log["mode"] != mode_tmp:
                        continue
                    epoch = log.pop('epoch')
                    if epoch not in log_dict:
                        log_dict[epoch] = defaultdict(list)
                    for k, v in log.items():
                        log_dict[epoch][k].append(v)
        log_dicts_collector = log_dicts_collector + log_dicts

    return log_dicts_collector


def load_json_logs_create_avg(json_logs, mode=None, special_term=None):
    """ Similar way to read and manipulate json file entries such as load_json_logs().
        This function filters non-relevant rows of the json file and computes mean and standard deviation
        over the loss values of the chosen mode. The dataframe for mean and std is then stored in separate json files.
        These json files can than be used in plot_curve() if std_dicts is not None.
        This way the standard deviation is plotted alongside the mean of each loss entry.
        By computing and storing the mean and std values in separate json files, the process from load_json_logs()
        and plot_curve() remains untouched, and it is easier to switch stds on and off.
        Also, the possibility to plot multiple graphs into one figure is preserved, with no adjustments.
        NOTE: this functionality is not provided for the metric 'mAP' only for 'loss'.

        Dataframes for mean and std are returned for checking.
    """
    if mode == "Train":
        list_of_modes = ["train"]
    elif mode == "Val":
        list_of_modes = ["val"]
    else:
        list_of_modes = ["train", "val"]
    for mode_tmp in list_of_modes:
        log_dicts = [dict() for _ in json_logs]
        df_averaging_collector = []
        for json_log, log_dict in zip(json_logs, log_dicts):
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
                df_averaging_collector.append(df_tmp)  # collects the dataframe from every new run
            # print("LEN", len(df_averaging_collector))
            # # Averaging process
            df_mean = (pd.concat(df_averaging_collector)
                         .groupby(['mode', 'epoch', 'iter'])
                         .agg(loss=('loss', 'mean')))
            df_std = (pd.concat(df_averaging_collector)
                        .groupby(['mode', 'epoch', 'iter'])
                        .agg(loss=('loss', 'std')))
            list_dict_mean = df_mean.reset_index().to_dict('records')
            list_dict_std = df_std.reset_index().to_dict('records')
            with open(f'None_Mean{special_term}.log.json', 'w') as fout:
                for dict_mean in list_dict_mean:
                    json.dump(dict_mean, fout)
                    fout.write('\n')
            with open(f'None_Std{special_term}.log.json', 'w') as fout:
                for dict_std in list_dict_std:
                    json.dump(dict_std, fout)
                    fout.write('\n')

    return df_mean, df_std


def main(args):

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    if args.mode is not None:
        if args.std_jsons is not None:
            std_json_logs = args.std_jsons
            assert len(json_logs) == len(std_json_logs)
            for std_json_log in std_json_logs:
                assert std_json_log.endswith('.json')
            log_dicts = load_json_logs(json_logs, mode=args.mode)
            std_dicts = load_json_logs(args.std_jsons, mode=args.mode)
        else:
            log_dicts = load_json_logs(json_logs, mode=args.mode)
            std_dicts = None
    else:
        log_dicts = load_json_logs(json_logs)
        std_dicts = None

    eval(args.task)(log_dicts, args, std_dicts)


if __name__ == '__main__':
    args = parse_args()
    main(args)
