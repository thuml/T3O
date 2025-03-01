import os

import matplotlib
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

from scipy.stats import boxcox
from scipy.special import inv_boxcox

# # 创建示例数据，例如正态分布数据
# np.random.seed(0)
# data = np.random.normal(loc=0, scale=1, size=100)

# 从ETTh1.csv中加载某个column的数据
from sklearn.preprocessing import MinMaxScaler

root_path = '../../_datasets/ts-data/ETT-small/'
data_path = 'ETTh1.csv'
df_raw = pd.read_csv(os.path.join(root_path, data_path))
# 丢掉第一列date
df_raw = df_raw.iloc[:, 1:]
# dict: column:np.1dArray
np_data_dict = {col: np.array(df_raw[col].values).reshape(-1) for col in df_raw.columns}
data = np_data_dict['OT'][0:50]


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None


def trans_1d_np_copy(data):
    data_np = np.array(data).copy()  # !!! copy
    assert data_np.ndim == 1, f'Invalid data shape: {data_np.shape}'
    return data_np


class Warper:
    def __init__(self, method):
        self.method = method
        # self.scaler = MinMaxScaler(feature_range=(-10, 10))  # FIXME：选择(-1,1)好像变化就无了因为近似y=x
        # self.scaler = None
        self.shift_value = None
        self.box_cox_lamda = None

    def pre_process(self, data):
        data = trans_1d_np_copy(data)
        if self.method == 'none':
            return data

        if self.method == 'log':
            # eps = 1e-8  # 防止除零错误
            # 平移到(1,inf)区间
            if min(data) <= 1:
                self.shift_value = 1 - min(data)
            else:
                self.shift_value = 1 - min(data)  # 最小值也放到1，log后到0
            data_shifted = data + self.shift_value
            assert np.min(data_shifted) > 0, f'Invalid data: {data_shifted}'
            data_log = np.log(data_shifted)
            res = data_log
        # elif self.method == 'box_cox':
        #     self.scaler = MinMaxScaler(feature_range=(0, 1))  # FIXME: 魔法数
        #     data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        #     # 获取最小值并计算平移常数
        #     eps = 1e-8  # 防止除零错误
        #     data_shifted = data_scaled + eps
        #     assert np.min(data_shifted) > 0, f'Invalid data: {data_shifted}'
        #     data_box_cox, self.box_cox_lamda = boxcox(data_shifted)
        #     res = data_box_cox
        # elif self.method == 'sigmoid':
        #     self.scaler = MinMaxScaler(feature_range=(-5, 5))  # FIXME: 魔法数
        #     data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        #     data_sigmoid = 1 / (1 + np.exp(-data_scaled))
        #     res = data_sigmoid
        # elif self.method == 'tanh':
        #     self.scaler = MinMaxScaler(feature_range=(-5, 5))  # FIXME: 魔法数
        #     data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        #     data_tanh = np.tanh(data_scaled)
        #     res = data_tanh
        else:
            raise Exception('Invalid warper method: {}'.format(self.method))
        assert res.ndim == 1, f'Invalid data shape: {res.shape}'
        assert np.isnan(res).sum() == 0 and np.isinf(res).sum() == 0, \
            f'Invalid data: {res}, method: {self.method}'
        return res

    def post_process(self, data):
        if self.method == 'none':
            return data

        if self.method == 'log':
            _data = np.exp(data)
            # 平移回去
            data_restored = _data - self.shift_value
        # elif self.method == 'box_cox':
        #     _data = inv_boxcox(data, self.box_cox_lamda)
        # elif self.method == 'sigmoid':
        #     eps = 1e-8
        #     data_sigmoid_inv = -np.log(1 / (data + eps) - 1)
        #     _data = data_sigmoid_inv
        # elif self.method == 'tanh':
        #     eps = 1e-8
        #     data_tanh_inv = np.arctanh(np.clip(data, -1 + eps, 1 - eps))
        #     _data = data_tanh_inv
        else:
            raise Exception('Invalid warper method: {}'.format(self.method))

        # 将数据从 [0, 1] 范围逆缩放到原始范围
        # data_restored = self.scaler.inverse_transform(_data.reshape(-1, 1)).flatten()

        return data_restored


# 实例化 Warper 类并选择方法
# methods = ['none', 'log', 'box_cox', 'sigmoid', 'tanh']

methods = ['none', 'log']
warper_results = {}

plt.figure(figsize=(16, 8))

for i, method in enumerate(methods):
    warper = Warper(method=method)
    data_transformed = warper.pre_process(data)
    data_restored = warper.post_process(data_transformed)
    warper_results[method] = {'transformed': data_transformed, 'restored': data_restored}

num_rows = 2
num_cols = (len(methods) * 2 + 1) // num_rows  # 计算列数

for i, method in enumerate(methods):
    for j in range(2):
        _data = warper_results[method]['transformed'] if j == 0 else warper_results[method]['restored']
        color = 'blue' if j == 1 else 'red'
        plt.subplot(num_rows, num_cols, i * 2 + j + 1)
        plt.plot(_data, label='{} {}'.format(method, 'transformed' if j == 0 else 'restored'), color=color)
        plt.legend()
        plt.title('Method: {} {}'.format(method, 'transformed' if j == 0 else 'restored'))
        plt.grid(True)

plt.tight_layout()
plt.show()
