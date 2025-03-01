import logging
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import inv_boxcox
from scipy.stats import boxcox, yeojohnson, boxcox_normmax
from torch.utils.data import DataLoader

from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import Weather, CustomDataset, Electricity, EttHour, get_dataset


class Warper:
    def __init__(self, method):
        self.method = method
        self.shift_values = None
        self.box_cox_lambda = None
        self.min_values = None
        self.max_values = None
        self.fail = False

    def pre_process(self, data):
        assert data.ndim == 3, f'Invalid data shape: {data.shape}'
        if self.method == 'none':
            return data

        batch_size, time_len, feature_dim = data.shape
        self.max_values = np.max(data, axis=1, keepdims=True)
        self.min_values = np.min(data, axis=1, keepdims=True)

        if self.method == 'log':
            self.shift_values = np.where(self.min_values <= 1, 1 - self.min_values, 0)
            data_shifted = data + self.shift_values
            res = np.log(data_shifted)

        elif self.method == 'sqrt':
            self.shift_values = np.where(self.min_values < 0, 1 - self.min_values, 0)
            data_shifted = data + self.shift_values
            res = np.sqrt(data_shifted)
        else:
            raise Exception('Invalid warper method: {}'.format(self.method))

        assert res.ndim == data.ndim, f'Invalid data shape: {res.shape}'
        assert np.isnan(res).sum() == 0 and np.isinf(res).sum() == 0, \
            f'Invalid data: {res}, method: {self.method}'

        if np.isnan(res).any() or np.isinf(res).any():
            logging.error(f"NaN or Inf values in transformed data: {res}")
            self.fail = True
            return data
        return res

    def post_process(self, data):
        if self.method == 'none':
            return data

        if self.fail:
            return data

        if self.method == 'log':
            _data = np.exp(data)
            data_restored = _data - self.shift_values

        elif self.method == 'sqrt':
            data_restored = np.square(data)
            data_restored = data_restored - self.shift_values

        else:
            raise Exception('Invalid warper method: {}'.format(self.method))

        if np.isnan(data_restored).any() or np.isinf(data_restored).any():
            logging.error(f"NaN or Inf values in transformed data: {data_restored}")
            data_restored = np.nan_to_num(data_restored, nan=0, posinf=0, neginf=0)
        return data_restored


# 假设 assert_timeseries_3d_np 是一个用于验证数据形状的函数
def assert_timeseries_3d_np(data):
    assert data.ndim == 3, f'Expected 3D data, got {data.ndim}D data'
    assert data.shape[2] == 1, f'Expected last dimension size 1, got {data.shape[2]}'


# 示例调用
history_seq = np.random.rand(3, 576, 1)  # 这是一个随机生成的示例数据
example = Warper('log')
preprocessed_data = example.pre_process(history_seq)
postprocessed_data = example.post_process(preprocessed_data)
print(preprocessed_data)
print(postprocessed_data)

seq_len = 96 * 6
pred_len = 96
dataset = get_dataset('Electricity')
# dataset = Electricity(root_path='../_datasets/ts-data/electricity/', data_path='electricity.csv')
# dataset = Weather(root_path='../_datasets/ts-data/weather/', data_path='weather.csv')
mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
    'test', 'OT', seq_len, Augmentor('none', False), 100, 100
custom_dataset = CustomDataset(dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample)
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
idx, history, label = next(iter(dataloader))  # batch, time, feature
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

# warper_methods = ['none', 'log', 'boxcox', 'yeojohnson']
warper_methods = ['none', 'log', 'sqrt']
fig, axs = plt.subplots(len(warper_methods), 1, figsize=(20, 20))
for i, warper_method in enumerate(warper_methods):
    warper = Warper(warper_method)
    preprocessed_data = warper.pre_process(history)
    postprocessed_data = warper.post_process(preprocessed_data)
    axs[i].plot(preprocessed_data[-1, :, 0], label=f'Pre-processed Data ({warper_method})', color='orange')
    axs[i].plot(postprocessed_data[-1, :, 0], label=f'Post-processed Data ({warper_method})', color='red')
    axs[i].set_title(f'Processed Data ({warper_method})')
    axs[i].legend()
plt.show()
