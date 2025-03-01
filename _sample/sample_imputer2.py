import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os

from AnyTransform.dataset import *


class Inputer:
    def __init__(self, detect_method, input_method, statistics_dict=None):
        self.detect_method = detect_method
        self.input_method = input_method
        self.statistics_dict = statistics_dict

    def pre_process(self, data):
        if self.detect_method == 'none':
            return data

        # 非常保守 因为可能数据偏移和存在趋势
        if self.detect_method == '3_sigma':
            fill_indices = self.detect_outliers_k_sigma(data, 3)
        elif self.detect_method == '5_sigma':
            fill_indices = self.detect_outliers_k_sigma(data, 5)
        elif self.detect_method == '7_sigma':
            fill_indices = self.detect_outliers_k_sigma(data, 7)
        elif self.detect_method == '3_iqr':
            fill_indices = self.detect_outliers_iqr(data, 3)
        elif self.detect_method == '5_iqr':
            fill_indices = self.detect_outliers_iqr(data, 5)
        elif self.detect_method == '7_iqr':
            fill_indices = self.detect_outliers_iqr(data, 7)
        else:
            raise ValueError(f"Unsupported detect method: {self.detect_method}")

        filled_data = self.fill_outliers(data, fill_indices)
        return filled_data

    def post_process(self, data):
        return data

    def detect_outliers_k_sigma(self, data, k_sigma):
        mean = self.statistics_dict['mean']
        std = self.statistics_dict['std']
        lower_bound = mean - k_sigma * std
        upper_bound = mean + k_sigma * std
        return np.where((data < lower_bound) | (data > upper_bound))[0]

    def detect_outliers_iqr(self, data, ratio):
        q1 = self.statistics_dict['q1']
        q3 = self.statistics_dict['q3']
        iqr = q3 - q1
        lower_bound = q1 - ratio * iqr
        upper_bound = q3 + ratio * iqr
        return np.where((data < lower_bound) | (data > upper_bound))[0]

    def fill_outliers(self, data, fill_indices):
        if self.input_method == 'none':
            return data

        filled_data = data.copy()
        if self.input_method == 'linear_interpolate':
            filled_data = self.linear_interpolate(filled_data, fill_indices)
        elif self.input_method == 'rolling_mean':
            filled_data = self.rolling_mean(filled_data, fill_indices)
        elif self.input_method == 'forward_fill':
            filled_data = self.forward_fill_zeros(filled_data, fill_indices)
        elif self.input_method == 'backward_fill':
            filled_data = self.backward_fill_zeros(filled_data, fill_indices)
        else:
            raise ValueError(f"Unsupported fill method: {self.input_method}")

        return filled_data

    def linear_interpolate(self, data, fill_indices):
        # data中非fill_indices为normal_indexes
        normal_indexes = np.setdiff1d(np.arange(len(data)), fill_indices)
        filled_data = data.copy()
        filled_data[fill_indices] = np.interp(fill_indices, normal_indexes, data[normal_indexes])
        return filled_data

    def rolling_mean(self, data, fill_indices):
        window_size = 1000  # FIXME: magic number
        filled_data = data.copy()
        window_size = min(max(window_size, len(fill_indices)), len(data) - 1)
        for idx in fill_indices:
            start = max(0, idx - window_size)
            end = min(len(data), idx + window_size + 1)
            neighbors = data[start:end]
            valid_neighbors = neighbors[neighbors != 0]
            if len(valid_neighbors) > 0:
                filled_data[idx] = np.mean(valid_neighbors)
        return filled_data

    def forward_fill_zeros(self, data, fill_indices):
        filled_data = data.copy()
        for idx in fill_indices:
            if idx > 0:
                filled_data[idx] = filled_data[idx - 1]
        return filled_data

    def backward_fill_zeros(self, data, fill_indices):
        filled_data = data.copy()
        for idx in fill_indices[::-1]:
            if idx < len(data) - 1:
                filled_data[idx] = filled_data[idx + 1]
        return filled_data


# 示例用法
# dataset = Weather(root_path='../../_datasets/ts-data/weather/', data_path='weather.csv', clean=False)
# dataset = EttHour(root_path='../../_datasets/ts-data/ETT-small/', data_path='ETTh1.csv')
# dataset = EttMinute(root_path='../../_datasets/ts-data/weather/', data_path='weather.csv')
dataset = Electricity(root_path='../../_datasets/ts-data/electricity/', data_path='electricity.csv')


# 计算统计量
train_data = dataset.np_data_dict['OT'][dataset.train_start:dataset.train_end]
statistics_dict = {
    'mean': np.mean(train_data),
    'std': np.std(train_data),
    'q1': np.percentile(train_data, 25),
    'q3': np.percentile(train_data, 75)
}

# 实例化Inputer
inputer = Inputer(detect_method='3_sigma', input_method='linear_interpolate', statistics_dict=statistics_dict)

# 获取插值前后的数据
train_data_original = train_data.copy()
train_data_filled = inputer.pre_process(train_data)

import matplotlib


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

# 绘制插值前后的数据
fig, axs = plt.subplots(2, 1, figsize=(20, 10))
axs[0].plot(train_data_original, label='Original Data')
axs[0].set_title('Original Data')
axs[1].plot(train_data_filled, label='Filled Data', color='orange')
axs[1].set_title('Filled Data')
plt.show()
