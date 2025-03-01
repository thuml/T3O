import logging

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader

from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import *
from AnyTransform.model import Uni2ts


class Inputer:
    def __init__(self, detect_method, fill_method, history_seq):
        # history_seq: (batch, time, feature)
        self.detect_method = detect_method
        self.fill_method = fill_method
        self.statistics_dict = self.get_statistics_dict(history_seq)

    def get_statistics_dict(self, history_seq):
        if self.detect_method == 'none' or self.fill_method == 'none':
            return None
        if 'sigma' in self.detect_method:
            mean = np.mean(history_seq, axis=(1), keepdims=True)
            std = np.std(history_seq, axis=1, keepdims=True)
            statistics_dict = {'mean': mean, 'std': std}
        elif 'iqr' in self.detect_method:
            q1 = np.percentile(history_seq, 25, axis=1, keepdims=True)
            q3 = np.percentile(history_seq, 75, axis=1, keepdims=True)
            statistics_dict = {'q1': q1, 'q3': q3}
        else:
            raise ValueError(f"Unsupported detect method: {self.detect_method}")
        return statistics_dict

    def pre_process(self, data):
        if self.detect_method == 'none' or self.fill_method == 'none':
            return data

        if 'sigma' in self.detect_method:
            k_sigma = int(self.detect_method.split('_')[0])
            fill_indices = self.detect_outliers_k_sigma(data, k_sigma)
        elif 'iqr' in self.detect_method:
            ratio = float(self.detect_method.split('_')[0])
            fill_indices = self.detect_outliers_iqr(data, ratio)
        else:
            raise ValueError(f"Unsupported detect method: {self.detect_method}")
        # logging.info(f"fill_indices: {fill_indices}")
        filled_data = self.fill_outliers(data, fill_indices)
        return filled_data

    def post_process(self, data):
        return data

    def detect_outliers_k_sigma(self, data, k_sigma):
        mean = self.statistics_dict['mean']
        std = self.statistics_dict['std']
        lower_bound = mean - k_sigma * std
        upper_bound = mean + k_sigma * std
        return np.where((data < lower_bound) | (data > upper_bound))

    def detect_outliers_iqr(self, data, ratio):
        q1 = self.statistics_dict['q1']
        q3 = self.statistics_dict['q3']
        iqr = q3 - q1
        lower_bound = q1 - ratio * iqr
        upper_bound = q3 + ratio * iqr
        return np.where((data < lower_bound) | (data > upper_bound))

    def fill_outliers(self, data, fill_indices):
        if self.detect_method == 'none' or self.fill_method == 'none':
            return data

        if self.fill_method == 'linear_interpolate':
            filled_data = self.linear_interpolate(data, fill_indices)
        elif self.fill_method == 'rolling_mean':
            filled_data = self.rolling_mean(data, fill_indices)
        elif self.fill_method == 'forward_fill':
            filled_data = self.forward_fill(data, fill_indices)
        elif self.fill_method == 'backward_fill':
            filled_data = self.backward_fill(data, fill_indices)
        else:
            raise ValueError(f"Unsupported fill method: {self.fill_method}")

        return filled_data

    def linear_interpolate(self, data, fill_indices):
        batch_size, seq_len, feature_dim = data.shape
        filled_data = data.copy()
        for b in range(batch_size):
            for f in range(feature_dim):
                # 使用布尔索引从 fill_indices[1] 中筛选出属于当前批次 b 的时间索引
                indices = fill_indices[1][fill_indices[0] == b]
                if len(indices) > 0:
                    normal_indices = np.setdiff1d(np.arange(seq_len), indices)
                    print(f"indices: {indices}")
                    print(f"normal_indices: {normal_indices}")
                    if len(normal_indices) == 0:
                        logging.warning(f"No normal indices for batch {b} feature {f} data: {data[b, :, f]}")
                        # 输出statistics_dict的值
                        for key, value in self.statistics_dict.items():
                            logging.warning(f"{key}: {value[b, 0, f]}")
                        continue
                    filled_data[b, indices, f] = np.interp(indices, normal_indices, data[b, normal_indices, f])
        return filled_data

    def rolling_mean(self, data, fill_indices):
        filled_data = data.copy()
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                window_size = len(indices) * 3  # FIXME: magic number
                for idx in indices:
                    start = max(0, idx - window_size)
                    end = min(seq_len, idx + window_size + 1)
                    neighbors = data[b, start:end, f]
                    # valid_neighbors = neighbors[neighbors != 0]
                    if len(neighbors) > 0:
                        filled_data[b, idx, f] = np.mean(neighbors)
        return filled_data

    def forward_fill(self, data, fill_indices):
        filled_data = data.copy()
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices:
                    if idx > 0:
                        filled_data[b, idx, f] = filled_data[b, idx - 1, f]
                    else:
                        filled_data[b, idx, f] = filled_data[b, idx + 1, f]
        return filled_data

    def backward_fill(self, data, fill_indices):
        filled_data = data.copy()
        batch_size, seq_len, feature_dim = data.shape
        for b in range(batch_size):
            for f in range(feature_dim):
                indices = fill_indices[1][fill_indices[0] == b]
                for idx in indices[::-1]:
                    if idx < seq_len - 1:
                        filled_data[b, idx, f] = filled_data[b, idx + 1, f]
                    else:
                        filled_data[b, idx, f] = filled_data[b, idx - 1, f]
        return filled_data


seq_len = 96 * 6
pred_len = 96
dataset = get_dataset('ETTh1')
mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
    'test', 'OT', seq_len, Augmentor('none', False), 100, 10
custom_dataset = CustomDataset(dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample)
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
iter_num = 5
iter = iter(dataloader)
for i in range(iter_num - 1):
    next(iter)
idx, history, label = next(iter)
scaler = dataset.get_train_scaler('standard', target_column)
history = scaler.transform(history.numpy().reshape(-1, 1)).reshape(batch_size, seq_len, 1)
label = scaler.transform(label.numpy().reshape(-1, 1)).reshape(batch_size, pred_len, 1)
seqs = history.copy()

import matplotlib


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

detect_methods = ['3_sigma', '5_sigma', '1.5_iqr', '3_iqr']
fill_methods = ['linear_interpolate', 'rolling_mean', 'forward_fill']

fig, axs = plt.subplots(len(detect_methods), len(fill_methods), figsize=(20, 20))
for i, detect_method in enumerate(detect_methods):
    for j, fill_method in enumerate(fill_methods):
        inputer = Inputer(detect_method=detect_method, fill_method=fill_method, history_seq=history)
        filled_data = inputer.pre_process(history)
        axs[i, j].plot(history[0, :, 0], label='Original Data')
        axs[i, j].plot(filled_data[0, :, 0], label=f'Filled Data ({detect_method}, {fill_method})', color='orange')
        axs[i, j].set_title(f'Filled Data ({detect_method}, {fill_method})')
plt.show()
