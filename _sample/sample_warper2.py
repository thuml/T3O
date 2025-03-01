import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import inv_boxcox
from scipy.stats import boxcox, yeojohnson, boxcox_normmax
from torch.utils.data import DataLoader

from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import Weather, CustomDataset, Electricity, EttHour


class Warper:
    def __init__(self, method):
        self.method = method
        self.shift_value = None
        self.box_cox_lambda = None

    def pre_process(self, data):
        assert_timeseries_3d_np(data)
        if self.method == 'none':
            return data

        if self.method == 'log':
            self.shift_value = 1 - np.min(data) if np.min(data) <= 1 else 0
            data_shifted = data + self.shift_value
            res = np.log(data_shifted)

        elif self.method == 'boxcox':
            self.shift_value = 1 - np.min(data) if np.min(data) <= 0 else 0
            data_shifted = data + self.shift_value
            self.box_cox_lambda = boxcox_normmax(data_shifted.flatten(), method='pearsonr')
            res = boxcox(data_shifted.flatten(), lmbda=self.box_cox_lambda)
            res = res.reshape(data.shape)

        elif self.method == 'yeojohnson':
            res, self.box_cox_lambda = yeojohnson(data.flatten())
            res = res.reshape(data.shape)

        else:
            raise Exception('Invalid warper method: {}'.format(self.method))

        assert res.ndim == data.ndim, f'Invalid data shape: {res.shape}'
        assert np.isnan(res).sum() == 0 and np.isinf(res).sum() == 0, \
            f'Invalid data: {res}, method: {self.method}'
        return res

    def post_process(self, data):
        if self.method == 'none':
            return data

        if self.method == 'log':
            _data = np.exp(data)
            data_restored = _data - self.shift_value

        elif self.method == 'boxcox':
            data_restored = inv_boxcox(data.flatten(), self.box_cox_lambda).reshape(data.shape)
            data_restored = data_restored - self.shift_value

        elif self.method == 'yeojohnson':
            # # Yeo-Johnson inverse transformation not directly available in scipy, so just return data
            # data_restored = data
            data_restored = self.inv_yeojohnson(data.flatten(), self.box_cox_lambda).reshape(data.shape)
        else:
            raise Exception('Invalid warper method: {}'.format(self.method))

        data_restored = np.nan_to_num(data_restored, nan=1e9, posinf=1e9, neginf=-1e9)
        return data_restored

    def inv_yeojohnson(self, y, lmbda):
        """Compute the inverse of the Yeo-Johnson transformation."""
        out = np.zeros_like(y)
        pos = y >= 0
        neg = ~pos
        if lmbda == 0:
            out[pos] = np.exp(y[pos]) - 1
            out[neg] = 1 - np.exp(-y[neg])
        else:
            out[pos] = np.power(y[pos] * lmbda + 1, 1 / lmbda) - 1
            out[neg] = 1 - np.power(-(y[neg] * lmbda - 1), 1 / lmbda)
        return out


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
dataset = EttHour(root_path='../_datasets/ts-data/ETT-small/', data_path='ETTh1.csv')
# dataset = Electricity(root_path='../_datasets/ts-data/electricity/', data_path='electricity.csv')
# dataset = Weather(root_path='../_datasets/ts-data/weather/', data_path='weather.csv')
mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
    'train', 'OT', seq_len, Augmentor('none', False), 3, 3
custom_dataset = CustomDataset(dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample)
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
history, label = next(iter(dataloader))  # batch, time, feature
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

warpper_methods = ['none', 'log', 'boxcox', 'yeojohnson']
fig, axs = plt.subplots(len(warpper_methods), 1, figsize=(20, 20))
for i, warper_method in enumerate(warpper_methods):
    warper = Warper(warper_method)
    preprocessed_data = warper.pre_process(history)
    postprocessed_data = warper.post_process(preprocessed_data)
    axs[i].plot(preprocessed_data[0, :, 0], label=f'Pre-processed Data ({warper_method})', color='orange')
    axs[i].plot(postprocessed_data[0, :, 0], label=f'Post-processed Data ({warper_method})', color='red')
    axs[i].set_title(f'Processed Data ({warper_method})')
    axs[i].legend()
plt.show()
