import os

import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def trans_1d_np(data):
    data_np = np.array(data)
    assert data_np.ndim == 1, f'Invalid data shape: {data_np.shape}'
    return data_np


class Inputer:
    def __init__(self, method='interpolation'):
        self.method = method

    def pre_process(self, data):
        data = trans_1d_np(data)
        if self.method == 'interpolation':
            filled_data = self.interpolate_zeros(data)
        elif self.method == 'rolling_mean':
            filled_data = self.rolling_mean_fill(data)
        # elif self.method == 'random_forest':
        #     filled_data = self.random_forest_fill(data)
        elif self.method == 'forward_fill':
            filled_data = self.forward_fill_zeros(data)
        elif self.method == 'backward_fill':
            filled_data = self.backward_fill_zeros(data)
        else:
            raise ValueError(f"Unsupported fill method: {self.method}")
        return filled_data

    def post_process(self, data):
        return data

    def interpolate_zeros(self, data):
        # Execute interpolation to fill consecutive zeros
        nonzero_indices = np.where(data != 0)[0]
        filled_data = data.copy()
        filled_data[data == 0] = np.interp(np.where(data == 0)[0], nonzero_indices, data[nonzero_indices])
        return filled_data

    def rolling_mean_fill(self, data):
        # Execute rolling mean fill for consecutive zeros
        window_size = 3
        filled_data = data.copy()
        zero_indices = np.where(data == 0)[0]
        window_size = min(max(window_size, len(zero_indices)), len(data) - 1)
        for idx in zero_indices:
            start = max(0, idx - window_size)
            end = min(len(data), idx + window_size + 1)
            neighbors = data[start:end]
            valid_neighbors = neighbors[neighbors != 0]
            if len(valid_neighbors) > 0:
                filled_data[idx] = np.mean(valid_neighbors)
        return filled_data

    def forward_fill_zeros(self, data):
        # Execute forward filling for consecutive zeros
        filled_data = data.copy()
        zero_indices = np.where(data == 0)[0]
        for idx in zero_indices:
            if idx > 0:
                filled_data[idx] = filled_data[idx - 1]
        return filled_data

    def backward_fill_zeros(self, data):
        # Execute backward filling for consecutive zeros
        filled_data = data.copy()
        zero_indices = np.where(data == 0)[0]
        # 倒过来遍历
        for idx in zero_indices[::-1]:
            if idx < len(data) - 1:
                filled_data[idx] = filled_data[idx + 1]
        return filled_data


# # 示例用法
# inputer = Inputer(method='interpolation')
# # inputer = Inputer(method='rolling_mean')
# test_data_list = [
#     [1, 2, 0, 0, 5, 0, 7, 0, 0, 10],
#     [1, 2, 0, 0, 5, 0, 7, 0, 0, 10, 0],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
#
#     [0, 0, 0, 4, 5, 6, 0, 0, 0, 0],
#     [1, 0, 0, 4, 5, 6, 0, 0, 0, 0],
#     [0, 0, 0, 4, 5, 6, 0, 0, 0, 10],
# ]
# for test_data in test_data_list:
#     filled_data = inputer.fill_zeros(test_data)
#     print(f'test_data: {test_data}, filled_data: {filled_data}')

method_name_list = ['interpolation', 'rolling_mean', 'forward_fill', 'backward_fill']
test_data_list = [
    [0, 0, 0, 4, 5, 6, 0, 0, 0, 0],
    [1, 0, 0, 4, 5, 6, 0, 0, 0, 0],
    [0, 0, 0, 4, 5, 6, 0, 0, 0, 10],

    [1, 2, 0, 0, 5, 0, 7, 0, 0, 10],
    [1, 2, 0, 0, 5, 0, 7, 0, 0, 10, 0],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
]

# hard_test_data = [24.37599945, 24.62800026, 24.125, 24.29299927, 26.3029995, 32.5019989,
#                   35.26599884, 34.84700012, 36.27199936, 34.34500122, 33.17200089, 30.74300003,
#                   28.98399925, 27.97900009, 26.05200005, 25.71699905, 25.8010006, 24.71199989,
#                   26.72200012, 26.63800049, 0, 0, 0, 0,
#                   0, 0, 0, 0, 0, 0,
#                   0, 34.68000031, 33.50699997, 33.50699997, 31.66399956, 29.06699944,
#                   25.88400078, 26.80599976, 32.25099945, 30.74300003, 36.68999863, 45.56999969,
#                   46.90999985, 45.31800079, 42.88899994, 47.99900055, 46.0719986, 48.5019989,
#                   48.66899872, 48.91999817]
# hard_test_data = [0, 0, 0, 0,
#                   0, 0, 0, 0, 0, 0,
#                   0, 34.68000031, 33.50699997, 33.50699997, 31.66399956, 29.06699944,
#                   25.88400078, 26.80599976, 32.25099945, 30.74300003, 36.68999863, 45.56999969,
#                   46.90999985, 45.31800079, 42.88899994, 47.99900055, 46.0719986, 48.5019989,
#                   48.66899872, 48.91999817]

hard_test_data = [24.37599945, 24.62800026, 24.125, 24.29299927, 26.3029995, 32.5019989,
                  35.26599884, 34.84700012, 36.27199936, 34.34500122, 33.17200089, 30.74300003,
                  28.98399925, 27.97900009, 26.05200005, 25.71699905, 25.8010006, 24.71199989,
                  26.72200012, 26.63800049, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, ]

for test_data in test_data_list:
    for method_name in method_name_list:
        inputer = Inputer(method=method_name)
        filled_data = inputer.pre_process(test_data)
        print(f'test_data: {test_data}, method: {method_name}, filled_data: {filled_data}')
    print()

# 在hard_test_data上补充缺失值并画图展示
import matplotlib.pyplot as plt


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

# 把 1原数据 2不同方法的插值后的数据 画在一张图上
plt.figure(figsize=(20, 10))
plt.plot(hard_test_data, label='original data', color='black')
for method_name in method_name_list:
    inputer = Inputer(method=method_name)
    filled_data = inputer.pre_process(hard_test_data)
    plt.plot(filled_data, label=f'{method_name} filled data')
plt.legend()
plt.show()
