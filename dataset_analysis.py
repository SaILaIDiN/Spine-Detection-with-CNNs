""" This script contains the analysis of the total dataset for geometrical and statistical properties.
Train, Val and Test will be investigated separately.
We create scatter plots of aspect ratio and area for gt boxes
We also plot the number of spines per image, number of images per stack
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_height_and_width(df):
    df_height = df["ymax"] - df["ymin"]
    df_width = df["xmax"] - df["xmin"]
    return df_height, df_width


def get_aspect_ratio_and_area(df_height, df_width):
    df_aspect_ratio = df_height/df_width
    df_area = df_height*df_width
    return df_aspect_ratio, df_area


def compute_mean_and_std(df):
    """ Input dataframe must have one column for ID and one for entries that should be averaged """
    mean = np.mean(df)
    std = np.std(df)
    return mean, std


def scatter_plot_h_w(X, Y, mean_and_std, xlabel, ylabel, title=None, output=None):
    """ Height and Width """
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axhline(y=mean_and_std["mean_height"], color="purple", linestyle="--", label="mean (heights)")
    plt.axvline(x=mean_and_std["mean_width"], color="red", linestyle="--", label="mean (widths)")
    plt.axhspan(ymin=mean_and_std["mean_height"]-mean_and_std["std_height"],
                ymax=mean_and_std["mean_height"]+mean_and_std["std_height"],
                alpha=0.1, color="purple", label="std (heights)")
    plt.axvspan(xmin=mean_and_std["mean_width"] - mean_and_std["std_width"],
                xmax=mean_and_std["mean_width"] + mean_and_std["std_width"],
                alpha=0.1, color="red", label="std (widths)")
    ax = plt.gca()
    ax.set_xlim(xmin=5, xmax=55)
    ax.set_ylim(bottom=5, ymax=55)
    plt.scatter(X, Y, linewidths=0.1, alpha=0.7)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0., fontsize="x-large")
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        print(f'save scatter plot to: {output}')
        plt.savefig(output)
        plt.cla()


def scatter_plot_a_a(X, Y, mean_and_std, xlabel, ylabel, title=None, output=None):
    """ Aspect Ratio and Area """
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axhline(y=mean_and_std["mean_aspect"], color="purple", linestyle="--", label="mean (aspect ratio)")
    plt.axvline(x=mean_and_std["mean_area"], color="red", linestyle="--", label="mean (area)")
    plt.axhspan(ymin=mean_and_std["mean_aspect"]-mean_and_std["std_aspect"],
                ymax=mean_and_std["mean_aspect"]+mean_and_std["std_aspect"],
                alpha=0.1, color="purple", label="std (aspect ratio)")
    plt.axvspan(xmin=mean_and_std["mean_area"] - mean_and_std["std_area"],
                xmax=mean_and_std["mean_area"] + mean_and_std["std_area"],
                alpha=0.1, color="red", label="std (area)")
    ax = plt.gca()
    ax.set_xlim(xmin=0, xmax=2300)
    ax.set_ylim(bottom=0, ymax=3.25)
    plt.scatter(X, Y, linewidths=0.1, alpha=0.7)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0., fontsize="x-large")
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        print(f'save scatter plot to: {output}')
        plt.savefig(output)
        plt.cla()


def histogram_plot(X, xlabel, ylabel, output=None):
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.locator_params(axis='x', nbins=5)  # only necessary for test_df
    bar_ticks = np.linspace(X.min(), X.max(), num=int((X.max()-X.min())/0.1)+1)
    plt.hist(X, rwidth=0.75, bins=bar_ticks)
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        print(f'save scatter plot to: {output}')
        plt.savefig(output)
        plt.cla()


def density_plot(X, xlabel, ylabel, output=None):
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.locator_params(axis='x', nbins=7)  # for train
    sns.distplot(X, kde=False, hist_kws={"rwidth": 0.75, 'edgecolor': 'black', 'alpha': 1.0}, bins=28)  # train

    # plt.locator_params(axis='x', nbins=6)  # for val
    # sns.distplot(X, kde=False, hist_kws={"rwidth": 0.75, 'edgecolor': 'black', 'alpha': 1.0}, bins=21)  # val

    # plt.locator_params(axis='x', nbins=5)  # for test
    # sns.distplot(X, kde=False, hist_kws={"rwidth": 0.75, 'edgecolor': 'black', 'alpha': 1.0}, bins=16)  # test

    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        print(f'save scatter plot to: {output}')
        plt.savefig(output)
        plt.cla()


if __name__ == "__main__":
    # we use the GT files for the tracking and evaluation part for train and val too, because then
    # the manipulation of dataframes is exactly the same for all three csv files
    train_csv_path = "output/tracking/GT/data_tracking_GT_train.csv"
    val_csv_path = "output/tracking/GT/data_tracking_GT_val.csv"
    test_csv_path = "output/tracking/GT/data_tracking_maj_wo_offset.csv"

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    # test_df = pd.read_csv(test_csv_path)
    test_df = pd.read_csv(test_csv_path).round(decimals=0)

    # # # Scatter Plot
    df_height_train, df_width_train = get_height_and_width(test_df)
    mean_height, std_height = compute_mean_and_std(df_height_train)
    mean_width, std_width = compute_mean_and_std(df_width_train)
    mean_and_std_dict = {"mean_height": mean_height, "mean_width": mean_width,
                         "std_height": std_height, "std_width": std_width}
    scatter_plot_h_w(df_width_train, df_height_train, mean_and_std_dict, "width [pixel]", "height [pixel]",
                     output="scatter_test")
    print(mean_height, std_height, mean_width, std_width)

    # # # Aspect Ratio Plot
    # df_height_train, df_width_train = get_height_and_width(val_df)
    # df_aspect_ratios, _ = get_aspect_ratio_and_area(df_height_train, df_width_train)
    # print(df_aspect_ratios)
    # print(df_aspect_ratios.round(decimals=1))
    # histogram_plot(df_aspect_ratios.round(decimals=2), "aspect ratio", "number of spines",
    #                "histogram_test")
    # density_plot(df_aspect_ratios.round(decimals=2), "aspect ratio", "number of spines",
    #              "density_val")

    # # # Aspect Ratio to Area Scatter Plot
    # df_height_train, df_width_train = get_height_and_width(train_df)
    # df_aspect_ratios, df_area = get_aspect_ratio_and_area(df_height_train, df_width_train)
    # mean_aspect, std_aspect = compute_mean_and_std(df_aspect_ratios)
    # mean_area, std_area = compute_mean_and_std(df_area)
    # mean_and_std_dict = {"mean_aspect": mean_aspect, "mean_area": mean_area,
    #                      "std_aspect": std_aspect, "std_area": std_area}
    # scatter_plot_a_a(df_area, df_aspect_ratios, mean_and_std_dict, "area [pixel]", "aspect ratio",
    #                  output="scatter_aspect_area_train")
    # print(mean_aspect, std_aspect, mean_area, std_area)
