import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


def plot_curve(log_dicts, args):
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
    if args.with_val_loss == "True":
        assert len(legend) == (len(args.json_logs) * 2 * len(args.keys))
        # because each json file creates two log_dict files one for train one for val
    else:
        assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            if args.with_val_loss:
                mode = 'val' if log_dict[1]['mode'][0] == 'val' else 'train'
                print(f'plot curve of {args.json_logs[min(i, abs(int(i-len(log_dicts)/2)))]}, '
                      f'mode is {mode}, metric is {metric}')
            else:
                print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
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
                # num_iters_per_epoch = log_dict[epochs[0]]['iter'][-2]
                num_iters_per_epoch = 844
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
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
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
                if args.ymax is not None:
                    ax.set_ylim(bottom=0, ymax=2.2)
                # ax.set_xlim(xmin=0)
                if args.xmargin is not None:
                    ax.margins(x=args.xmargin, y=0.1)
            plt.legend()
        if args.title is not None:
            plt.title(f"{args.title.split('_')[0]}, {args.title.split('_')[-1]}", fontsize=15)
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
        help='path of train log in json format')
    parser_plt.add_argument(
        '--with_val_loss',
        type=str,
        default=None,
        help='activates the search for val loss in the JSON log file and the separate plotting'
    )
    parser_plt.add_argument(
        '--xmargin',
        type=float,
        default=None,
        help='custom change of margin of x-axis, element [0, 1]'
    )
    parser_plt.add_argument(
        '--ymax',
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


def load_json_logs(json_logs, with_val_loss=None):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts_collector = []
    list_of_modes = ["train"] if with_val_loss is None else ["train", "val"]
    for mode in list_of_modes:
        log_dicts = [dict() for _ in json_logs]
        for json_log, log_dict in zip(json_logs, log_dicts):
            with open(json_log, 'r') as log_file:
                for line in log_file:
                    log = json.loads(line.strip())
                    # skip lines without `epoch` field
                    if 'epoch' not in log:
                        continue
                    if log["mode"] == mode:
                        continue
                    epoch = log.pop('epoch')
                    if epoch not in log_dict:
                        log_dict[epoch] = defaultdict(list)
                    for k, v in log.items():
                        log_dict[epoch][k].append(v)
        log_dicts_collector = log_dicts_collector + log_dicts

    return log_dicts_collector


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    if args.with_val_loss == "True":
        log_dicts = load_json_logs(json_logs, with_val_loss=args.with_val_loss)
    else:
        log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()
