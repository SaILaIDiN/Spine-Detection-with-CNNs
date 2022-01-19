""" This script contains the first analysis of the dataset before analysing the spine geometry.
First, we get the number of images in training, val and test.
Second, we get the number of stacks in training, val and test.
Third, we get the number of images per stack as a distribution for train, val and test.
Fourth, we get the number of spines per image as a distribution for train, val and test.
"""
import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt


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
    print(len(all_matches))
    img_list_unique = np.unique(all_matches)
    return len(img_list_unique)


def number_of_stacks(dir_csv):
    """ Is only correct if we assume, that the image name is unique over all labeling persons. """
    df = pd.read_csv(dir_csv)
    img_list = list(sorted(df["filename"].values.tolist()))
    re_pattern = re.compile(r'(.*)day(\d{1,2})stack(\d{1,2})')
    all_matches = []
    for img in img_list:
        matches = re_pattern.finditer(img)
        for match in matches:
            all_matches.append(match.group(0))
    print(len(all_matches))
    img_list_unique = np.unique(all_matches)
    return len(img_list_unique)


def size_of_stacks(dir_csv):
    """ """
    df = pd.read_csv(dir_csv)
    img_list = list(sorted(df["filename"].values.tolist()))
    re_pattern_img = re.compile(r'(.*)day(\d{1,2})stack(\d{1,2})-(\d{2})')  # get filenames only up to the stack number
    all_matches = []  # will contain a list of all filenames
    for img in img_list:
        matches = re_pattern_img.finditer(img)
        for match in matches:
            all_matches.append(match.group(0))
    img_list_unique = np.unique(all_matches)  # make each unique filename appear only once
    re_pattern_stack = re.compile(r'(.*)day(\d{1,2})stack(\d{1,2})')
    all_matches = []  # will contain a list of all filenames up to the stack part
    for img in img_list_unique:
        matches = re_pattern_stack.finditer(img)
        for match in matches:
            all_matches.append(match.group(0))
    # img_list_unique = np.unique(all_matches)  # make each unique stack appear only once
    dist_of_unique_stacks = Counter(all_matches)
    return dist_of_unique_stacks


def number_of_spines_per_image(dir_csv):
    """ """
    df = pd.read_csv(dir_csv)
    img_list = list(sorted(df["filename"].values.tolist()))
    dist_spines_by_image = Counter(img_list)
    return dist_spines_by_image


def hist_plot_stacks(X, xlabel, ylabel, output=None):
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax = plt.gca()

    ax.set_ylim(bottom=0, ymax=4.5)  # for train
    plt.locator_params(axis='x', nbins=10)  # for train
    plt.locator_params(axis='y', nbins=5)  # for train
    plt.hist(X, rwidth=0.75, edgecolor='black', alpha=1.0, bins=63, align='left')  # train

    # ax.set_ylim(bottom=0, ymax=2.5)  # for val
    # plt.locator_params(axis='x', nbins=14)  # for val
    # plt.locator_params(axis='y', nbins=3)  # for val
    # plt.hist(X, rwidth=0.75, edgecolor='black', alpha=1.0, bins=15, align='left')  # val

    # ax.set_ylim(bottom=0, ymax=2.5)  # for test
    # plt.locator_params(axis='x', nbins=9)  # for test
    # plt.locator_params(axis='y', nbins=3)  # for test
    # plt.hist(X, rwidth=0.75, edgecolor='black', alpha=1.0, bins=8, align='left')  # test

    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        print(f'save scatter plot to: {output}')
        plt.savefig(output)
        plt.cla()


def hist_plot_spines(X, xlabel, ylabel, output=None):
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.locator_params(axis='x', nbins=10)  # for train
    plt.locator_params(axis='y', nbins=5)  # for train
    plt.hist(X, rwidth=0.75, edgecolor='black', alpha=1.0, bins=25, align='left')  # train

    # plt.locator_params(axis='x', nbins=10)  # for val
    # plt.locator_params(axis='y', nbins=4)  # for val
    # plt.hist(X, rwidth=0.75, edgecolor='black', alpha=1.0, bins=14, align='left')  # val

    # plt.locator_params(axis='x', nbins=10)  # for test
    # plt.locator_params(axis='y', nbins=6)  # for test
    # plt.hist(X, rwidth=0.75, edgecolor='black', alpha=1.0, bins=18, align='left')  # test

    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        print(f'save scatter plot to: {output}')
        plt.savefig(output)
        plt.cla()


if __name__ == '__main__':
    train_csv_path = "output/tracking/GT/data_tracking_GT_train.csv"
    val_csv_path = "output/tracking/GT/data_tracking_GT_val.csv"
    test_csv_path = "output/tracking/GT/data_tracking_maj_wo_offset.csv"

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # # # 1
    img_list_train = list(sorted(train_df["filename"].values.tolist()))
    img_list_val = list(sorted(val_df["filename"].values.tolist()))
    img_list_test = list(sorted(test_df["filename"].values.tolist()))
    # make them unique
    img_list_train = np.unique(img_list_train)
    img_list_val = np.unique(img_list_val)
    img_list_test = np.unique(img_list_test)
    print(len(img_list_train))
    print(len(img_list_val))
    print(len(img_list_test))

    # # # 2
    n_images_train = length_of_train_set(train_csv_path)
    n_stacks_train = number_of_stacks(train_csv_path)
    n_images_val = length_of_train_set(val_csv_path)
    n_stacks_val = number_of_stacks(val_csv_path)
    n_images_test = length_of_train_set(test_csv_path)
    n_stacks_test = number_of_stacks(test_csv_path)
    print(n_images_train)
    print(n_stacks_train)
    print(n_images_val)
    print(n_stacks_val)
    print(n_images_test)
    print(n_stacks_test)

    # # # 3
    dist_stacks_train = size_of_stacks(train_csv_path)
    dist_stacks_val = size_of_stacks(val_csv_path)
    dist_stacks_test = size_of_stacks(test_csv_path)
    # hist_plot_stacks(X=np.array(list(dist_stacks_val.values())).astype(float),
    #                  xlabel="stack size", ylabel="count", output="hist_stack_size_val")

    # # # 4
    dist_spines_train = number_of_spines_per_image(train_csv_path)
    dist_spines_val = number_of_spines_per_image(val_csv_path)
    dist_spines_test = number_of_spines_per_image(test_csv_path)
    # hist_plot_spines(X=np.array(list(dist_spines_test.values())).astype(float),
    #                  xlabel="spines per image", ylabel="count", output="hist_spine_number_test")
