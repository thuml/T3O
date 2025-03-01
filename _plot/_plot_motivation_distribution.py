import logging
import os
from datetime import datetime
from math import ceil

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from torch.utils.data import DataLoader

from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import get_dataset, CustomDataset
from AnyTransform.model import get_model

import matplotlib

from AnyTransform.pipeline import adaptive_infer
from AnyTransform.utils import get_params_space_and_org, set_seed


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

plot_count = 0


def plot_histogram_once(data_name, model_name, pred_len):
    logging.info(f"Begin to plot histograms ...")

    mode = 'test'
    metric_names = ['mse', 'mae']

    parent_dir_name = '/Users/cenzhiyao/PycharmProjects/ts_adaptive_inference/new_moti'
    child_dir_name = '240715-003836-P500-S500-ABnone-SD0'
    res_file = os.path.join(parent_dir_name, child_dir_name, data_name, model_name, f'pred_len-{pred_len}', 'OT',
                            'pd_data.csv')
    pd_data = pd.read_csv(res_file)

    exp_res_file = os.path.join(parent_dir_name, child_dir_name, '_experiment_status.csv')
    pd_exp_data = pd.read_csv(exp_res_file)
    # 筛选：data_name, model_name, pred_len
    pd_exp_data = pd_exp_data[(pd_exp_data['data_name'] == data_name) & (pd_exp_data['pred_len'] == pred_len) &
                              (pd_exp_data['model_name'] == model_name)]
    params_space, origin_param_dict = get_params_space_and_org(fast_mode=True)
    params_keys = origin_param_dict.keys()
    our_param_dict = pd_exp_data[params_keys].iloc[0].to_dict()

    mode_data = pd_data[pd_data['mode'] == mode]
    mask_origin = np.logical_and.reduce([mode_data[key] == value for key, value in origin_param_dict.items()])
    mask_val_top1 = np.logical_and.reduce([mode_data[key] == value for key, value in our_param_dict.items()])

    for metric_name in metric_names:
        logging.info(f"Plotting histogram for {metric_name}...")
        origin_metric_values = mode_data[mask_origin][metric_name].values
        val_top1_metric_values = mode_data[mask_val_top1][metric_name].values

        assert len(origin_metric_values) == len(val_top1_metric_values), \
            f"Lengths do not match: {len(origin_metric_values)} != {len(val_top1_metric_values)}"

        plt.figure(figsize=figsize)

        # # Plot histograms for origin and val_top1 metric values
        # plt.hist(origin_metric_values, bins=50, alpha=0.5, label='Origin', color='blue', log=True)
        # plt.hist(val_top1_metric_values, bins=50, alpha=0.5, label='Val Top1', color='orange', log=True)

        # Compute histograms
        bins = np.linspace(min(origin_metric_values.min(), val_top1_metric_values.min()),
                           max(origin_metric_values.max(), val_top1_metric_values.max()), 50)
        origin_hist, bins = np.histogram(origin_metric_values, bins=bins)
        val_top1_hist, _ = np.histogram(val_top1_metric_values, bins=bins)
        # Plot histograms with same bar width
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = (bins[1] - bins[0]) * 0.5  # Set bar width to 40% of bin width # 中间白色太亮眼 0.4 0.5 1
        plt.bar(bin_centers - width / 2 * 1, origin_hist, width=width, label='Vanilla', color='orange')
        plt.bar(bin_centers + width / 2 * 1, val_top1_hist, width=width, label='Our', color='blue')
        # ...
        # bin_centers = (bins[:-1] + bins[1:]) / 2
        # width = (bins[1] - bins[0]) * 0.5  # Set bar width to 40% of bin width # 中间白色太亮眼 0.4 0.5 1
        # plt.bar(bin_centers - width / 2 * 0, origin_hist, width=width, label='Original Error', color='orange', alpha=0.5)
        # plt.bar(bin_centers + width / 2 * 0, val_top1_hist, width=width, label='Our Error', color='blue', alpha=0.5)
        # FIXME: log or not
        # plt.xscale('log') # bin宽度形状会变得不一致
        # plt.yscale('log') # 面积比例会不一致？
        # 看不出来明显优势？？？ -》 xlog才能显示出大小值的差异？
        # plt.bar(bin_centers - width / 2, origin_hist, width=width, alpha=0.5, label='Origin', color='blue')
        # plt.bar(bin_centers + width / 2, val_top1_hist, width=width, alpha=0.5, label='Val Top1', color='orange')

        xlabel = f'Mean Absolute Error (MAE)' if metric_name == 'mae' else f'Mean Squared Error (MSE)'
        plt.xlabel(f'{xlabel}', fontsize=fontsize)
        plt.ylabel('Number of Predictions', fontsize=fontsize)
        plt.title(f'Error Metric Distribution Over Predictions', fontsize=fontsize)
        plt.legend(loc='upper right', fontsize=fontsize)

        date_time_str = datetime.now().strftime("%y%m%d-%H%M%S")
        plot_path = f'/Users/cenzhiyao/Desktop/images/motivation-histogram/motivation-{date_time_str}' \
                    f'_{data_name}_{model_name}_{pred_len}_{metric_name}.png'
        plt.savefig(plot_path)
        logging.info(f"Histogram saved to {plot_path}")
        plt.close()


if __name__ == "__main__":
    print("start")
    set_seed(0)

    figsize = (10, 5)
    fontsize = 20

    # kwargs = {'data_name': 'Exchange', 'model_name': 'Chronos-tiny', 'pred_len': 96}
    # plot_histogram_once(**kwargs)

    # kwargs = {'data_name': 'Exchange', 'model_name': 'Timer-LOTSA', 'pred_len': 24} # 幅度太大有点假？
    # plot_histogram_once(**kwargs)

    # kwargs = {'data_name': 'Exchange', 'model_name': 'Timer-UTSD', 'pred_len': 24}  # 还行 左侧优势不太明显
    # plot_histogram_once(**kwargs)

    kwargs = {'data_name': 'Exchange', 'model_name': 'Timer-LOTSA', 'pred_len': 96}  # 可以！
    plot_histogram_once(**kwargs)

    # kwargs = {'data_name': 'Exchange', 'model_name': 'Timer-LOTSA', 'pred_len': 192}  # 原始分布有些太离散了
    # plot_histogram_once(**kwargs)
