import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import get_dataset, CustomDataset
from AnyTransform.model import get_model
from AnyTransform.transforms import Aligner, Normalizer

import numpy as np

from AnyTransform.utils import my_clip


class Differentiator:
    def __init__(self, n, correct_flag=False):
        self.n = n
        self.history_diff_data = []
        self.diff_data = None
        self.data_in = None
        self.correct_flag = correct_flag

    def pre_process(self, data):
        if self.n == 0:
            return data

        batch, time, feature = data.shape
        self.data_in = data

        self.history_diff_data = []
        diff_data = data

        for _ in range(self.n):
            self.history_diff_data.append(diff_data[:, 0:1, :])  # 记录差分前的第一个值
            diff_data = np.diff(diff_data, axis=1)
        self.diff_data = diff_data

        aligner = Aligner(time, 'zero_pad')  # FIXME
        res = aligner.pre_process(diff_data)
        return res

    def post_process(self, data):
        if self.n == 0:
            return data

        batch, time, feature = data.shape

        inv_diff_data_total = np.concatenate([self.diff_data, data], axis=1)
        for i in range(self.n - 1, -1, -1):
            # FIXME 合并之前调整inv_diff_data_total的均值和self.history_diff_data[i]一致???
            inv_diff_data_total = np.concatenate([self.history_diff_data[i], inv_diff_data_total], axis=1)
            inv_diff_data_total = np.cumsum(inv_diff_data_total, axis=1)
            inv_diff_data_total = inv_diff_data_total[:, time1:, :]

        pre_time = self.diff_data.shape[1]
        assert pre_time + time + self.n == inv_diff_data_total.shape[1], \
            f"{pre_time} + {self.n} + {time} != {inv_diff_data_total.shape[1]}"
        inv_diff_data = inv_diff_data_total[:, pre_time:pre_time + time, :]

        res = inv_diff_data
        res = my_clip(self.data_in, res, nan_inf_clip_factor=5, min_max_clip_factor=5)
        return res


# 定义参数
seq_len = 96 * 6
pred_len = 96 * 6

# 获取数据集
dataset = get_dataset('Weather')
mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
    'train', 'OT', seq_len, Augmentor('none', 'fix'), 10, 10
custom_dataset = CustomDataset(dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample)
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
idx, history, label = next(iter(dataloader))  # batch, time, feature

history = history.reshape(-1, seq_len, 1).numpy()
label = label.reshape(-1, pred_len, 1).numpy()

# 缩放数据
history_transformed = np.zeros_like(history)
label_transformed = np.zeros_like(label)
scalers = []

for i in range(batch_size):
    scaler = StandardScaler()
    history_batch = history[i].reshape(-1, 1)
    label_batch = label[i].reshape(-1, 1)

    # 对每个 batch 的数据进行缩放
    history_transformed[i] = scaler.fit_transform(history_batch).reshape(seq_len, 1)
    label_transformed[i] = scaler.transform(label_batch).reshape(pred_len, 1)

    scalers.append(scaler)  # 保存每个 batch 的 scaler 以备后用

# 将数据转换回原始类型
history = np.array(history_transformed)
label = np.array(label_transformed)

import matplotlib


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

# 定义模型
model = get_model('Timer-UTSD', 'cpu')

# 定义不同的 n 阶差分器
n_orders = [0, 1]  # 2就不太行了
correct_flags = [False, True]

fig, axs = plt.subplots(len(n_orders), len(correct_flags), figsize=(20, 20))
for i, n_order in enumerate(n_orders):
    for j, correct_flag in enumerate(correct_flags):
        print(f"Order={n_order}, Correct Flag={correct_flag}")
        differentiator = Differentiator(n=n_order)
        data_after_differentiator = differentiator.pre_process(history)
        normalizer = Normalizer('standard', 'input', data_after_differentiator, None, None)
        data_after_normalizer = normalizer.pre_process(data_after_differentiator)

        preds = model.forcast(data_after_normalizer, pred_len)

        preds_after_normalizer = normalizer.post_process(preds)
        preds_after_differentiator = differentiator.post_process(preds_after_normalizer)
        postprocessed_data = preds_after_differentiator

        truth_total = np.concatenate([history, label], axis=1)

        axs[i, j].plot(np.arange(seq_len + pred_len), truth_total[0, :, 0], label='Original Data', color='blue')
        axs[i, j].plot(np.arange(seq_len, seq_len + pred_len), postprocessed_data[0, :, 0], label='Predicted Data',
                       color='orange')

        axs[i, j].set_title(f'{n_order}-Order Differentiation, Correct Flag={correct_flag}')
        axs[i, j].legend()

# plt.tight_layout()
plt.show()

# fig, axs = plt.subplots(len(n_orders), 1, figsize=(10, 6 * len(n_orders)))
#
# for i, n_order in enumerate(n_orders):
#     differentiator = Differentiator(n=n_order)
#     data_after_differentiator = differentiator.pre_process(history)
#     normalizer = Normalizer('standard', 'input', data_after_differentiator, None, None)
#     data_after_normalizer = normalizer.pre_process(data_after_differentiator)
#
#     preds = model.forcast(data_after_normalizer, pred_len)
#
#     preds_after_normalizer = normalizer.post_process(preds)
#     preds_after_differentiator = differentiator.post_process(preds_after_normalizer)
#     postprocessed_data = preds_after_differentiator
#
#     # 可视化结果
#     truth_total = np.concatenate([history, label], axis=1)
#     axs[i].plot(np.arange(seq_len + pred_len), truth_total[0, :, 0], label='Original Data', color='blue')
#     axs[i].plot(np.arange(seq_len, seq_len + pred_len), postprocessed_data[0, :, 0], label='Predicted Data',
#                 color='orange')
#     axs[i].set_title(f'{n_order}-Order Differentiation')
#     axs[i].legend()
#
# plt.tight_layout()
# plt.show()
