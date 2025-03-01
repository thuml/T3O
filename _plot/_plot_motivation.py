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


def plot_once(data_name, model_name, split_idx):
    print(f"plot_once: {data_name}, {model_name}, {split_idx}")
    global plot_count

    patch_len = 96
    seq_len = patch_len * 15
    # pred_len = patch_len // 2
    pred_len = patch_len * 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_name, device)
    dataset = get_dataset(data_name)
    mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
        'test', 'OT', seq_len, Augmentor('none', 'fix', pred_len), 100000, 1000
    custom_dataset = CustomDataset(dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
    data_iter = iter(dataloader)
    while True:
        idxes, aug_methods, history, label = next(data_iter)
        # print(idxes)
        if split_idx in list(idxes):
            print("found")
            idxes_idx = list(idxes).index(split_idx)
            historys = history.reshape(-1, seq_len, 1).numpy()[idxes_idx:idxes_idx + 1]
            labels = label.reshape(-1, pred_len, 1).numpy()[idxes_idx:idxes_idx + 1]
            break

    params_space, origin_param_dict = get_params_space_and_org(fast_mode=True)
    org_kwargs = origin_param_dict.copy()
    org_kwargs.update({'history_seqs': historys, 'model': model, 'dataset': dataset,
                       'target_column': target_column, 'patch_len': patch_len, 'pred_len': pred_len, 'mode': mode})
    org_preds, _, _ = adaptive_infer(**org_kwargs)

    parent_dir_name = '/Users/cenzhiyao/PycharmProjects/ts_adaptive_inference/new_moti'
    child_dir_name = '240715-003836-P500-S500-ABnone-SD0'
    res_file = os.path.join(parent_dir_name, child_dir_name, '_experiment_status.csv')
    data_pd = pd.read_csv(res_file)
    # 筛选：data_name, model_name, pred_len
    data_pd = data_pd[(data_pd['data_name'] == data_name) & (data_pd['pred_len'] == pred_len) &
                      (data_pd['model_name'] == model_name)]
    params_keys = origin_param_dict.keys()
    our_param_dict = data_pd[params_keys].iloc[0].to_dict()
    print(f'our_param_dict: {our_param_dict}')
    our_param_dict.update({'history_seqs': historys, 'model': model, 'dataset': dataset,
                           'target_column': target_column, 'patch_len': patch_len, 'pred_len': pred_len, 'mode': mode})
    our_preds, _, _ = adaptive_infer(**our_param_dict)

    history_line = historys[0, int(-1.5 * pred_len):, 0]
    label_line = labels[0, :, 0]
    ground_truth = np.concatenate([history_line, label_line])
    ground_truth_x = np.arange(len(ground_truth))

    org_pred_line = np.concatenate([history_line, org_preds[0, :, 0]])
    our_pred_line = np.concatenate([history_line, our_preds[0, :, 0]])

    # plt.figure(figsize=(ceil(len(label_line) / patch_len) * 5, 5))
    # plt.plot(ground_truth_x, our_pred_line, label="Our Prediction", color='blue', linestyle='-', linewidth=2)
    # plt.plot(ground_truth_x, org_pred_line, label="Original Prediction", color='blue', linestyle='--', linewidth=1)
    # plt.plot(ground_truth_x, ground_truth, label="Ground Truth", color='orange', linestyle='-', linewidth=2)
    # plt.legend(loc='upper left', fontsize=20) if plot_count == 0 else None
    # plt.title(f'{model_name} on {data_name}', fontsize=20)
    # date_time_str = datetime.now().strftime("%y%m%d-%H%M%S")
    # plt.savefig(f'/Users/cenzhiyao/Desktop/images/motivation/motivation-{date_time_str}'
    #             f'_{data_name}_{model_name}_{split_idx}.png')
    # plot_count += 1

    fontsize = 50
    length, width = ceil(len(label_line) / patch_len) * 5, 4

    plt.figure(figsize=(length, width))
    plt.plot(ground_truth_x, org_pred_line, label="Original Prediction", color='blue', linestyle='--', linewidth=1)
    plt.plot(ground_truth_x, ground_truth, label="Ground Truth", color='orange', linestyle='-', linewidth=2)
    # plt.legend(loc='upper left', fontsize=fontsize) if plot_count == 0 else None
    # plt.title(f'{model_name} on {data_name}', fontsize=fontsize)
    date_time_str = datetime.now().strftime("%y%m%d-%H%M%S")
    plt.savefig(f'/Users/cenzhiyao/Desktop/images/motivation/motivation-{date_time_str}'
                f'_{data_name}_{model_name}_{split_idx}_org.png')

    plt.figure(figsize=(length, width))
    plt.plot(ground_truth_x, our_pred_line, label="Our Prediction", color='blue', linestyle='-', linewidth=2)
    plt.plot(ground_truth_x, ground_truth, label="Ground Truth", color='orange', linestyle='-', linewidth=2)
    # plt.legend(loc='upper left', fontsize=fontsize) if plot_count == 0 else None
    # plt.title(f'{model_name} on {data_name}', fontsize=fontsize)
    date_time_str = datetime.now().strftime("%y%m%d-%H%M%S")
    plt.savefig(f'/Users/cenzhiyao/Desktop/images/motivation/motivation-{date_time_str}'
                f'_{data_name}_{model_name}_{split_idx}_our.png')

    # plt.figure(figsize=(length, width))
    # plt.plot(our_pred_line[-len(label_line):], label="Our Prediction", color='blue', linestyle='-', linewidth=2)
    # plt.plot(label_line, label="Ground Truth", color='orange', linestyle='-', linewidth=2)
    # # plt.legend(loc='upper left', fontsize=fontsize) if plot_count == 0 else None
    # # plt.title(f'{model_name} on {data_name}', fontsize=fontsize)
    # date_time_str = datetime.now().strftime("%y%m%d-%H%M%S")
    # plt.savefig(f'/Users/cenzhiyao/Desktop/images/motivation/motivation-{date_time_str}'
    #             f'_{data_name}_{model_name}_{split_idx}_our.png')

    plot_count += 1


if __name__ == "__main__":
    print("start")
    set_seed(0)

    kwargs = {'data_name': 'ETTm2', 'model_name': 'MOIRAI-large', 'split_idx': 49956}  # context + sample
    plot_once(**kwargs)
    kwargs = {'data_name': 'Weather', 'model_name': 'Timer-UTSD', 'split_idx': 45354}  # anomaly detection
    plot_once(**kwargs)
    kwargs = {'data_name': 'Exchange', 'model_name': 'Chronos-tiny',
              'split_idx': 6835}  # differentiator 6985 6785! Timer-UTSD 6735 6835 Chronos-tiny
    plot_once(**kwargs)

    # waiting
    # kwargs = {'data_name': 'ETTm2', 'model_name': 'Timer-LOTSA', 'split_idx': 49956}  # long context
    # plot_once(**kwargs)

    # 废弃
    # kwargs = {'data_name': 'Weather', 'model_name': 'Timer-UTSD', 'split_idx': 45210} # 差异不明显
    # plot_once(**kwargs)
    # kwargs = {'data_name': 'Weather', 'model_name': 'Chronos-tiny', 'split_idx': 45354} # 不好
    # plot_once(**kwargs)
    # kwargs = {'data_name': 'Weather', 'model_name': 'Chronos-tiny', 'split_idx': 45210}  # 不好
    # plot_once(**kwargs)
    # kwargs = {'data_name': 'Traffic', 'model_name': 'Timer-UTSD', 'split_idx': 15832} # 差异不明显
    # plot_once(**kwargs)

    # 45210
    # traffic 15832

    # kwargs = {'data_name': 'ETTm2', 'model_name': 'MOIRAI-small', 'split_idx': 45986}
    # plot_once(**kwargs)

    # 263499.pdf
