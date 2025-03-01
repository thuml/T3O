import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import get_dataset, CustomDataset


class Denoiser:
    def __init__(self, method):
        assert method in ['none', 'moving_average', 'ewma', 'median']
        self.method = method
        self.window_size = 3
        self.alpha = 0.1

    def pre_process(self, data):
        if self.method == 'none':
            return data

        if self.method == 'moving_average':
            return self.moving_average(data)
        elif self.method == 'ewma':
            return self.ewma(data)
        elif self.method == 'median':
            return self.median_filter(data)
        # elif self.method == 'low_pass':
        #     return self.low_pass_filter(data)
        else:
            raise ValueError(f"Unsupported denoise method: {self.method}")

    def moving_average(self, data):
        window_size = self.window_size
        batch, time, feature = data.shape
        smoothed_data = np.zeros_like(data)
        for b in range(batch):
            for f in range(feature):
                smoothed_data[b, :, f] = np.convolve(data[b, :, f], np.ones(window_size) / window_size, mode='same')
        return smoothed_data

    def ewma(self, data):
        alpha = self.alpha
        batch, time, feature = data.shape
        smoothed_data = np.zeros_like(data)
        for b in range(batch):
            for f in range(feature):
                for t in range(1, time):
                    smoothed_data[b, t, f] = alpha * data[b, t, f] + (1 - alpha) * smoothed_data[b, t - 1, f]
        return smoothed_data

    def median_filter(self, data):
        window_size = self.window_size
        batch, time, feature = data.shape
        smoothed_data = np.zeros_like(data)
        for b in range(batch):
            for f in range(feature):
                smoothed_data[b, :, f] = self._apply_median_filter(data[b, :, f], window_size)
        return smoothed_data

    def _apply_median_filter(self, data, window_size):
        pad_size = window_size // 2
        padded_data = np.pad(data, pad_size, mode='edge')
        smoothed_data = np.zeros_like(data)
        for i in range(len(data)):
            smoothed_data[i] = np.median(padded_data[i:i + window_size])
        return smoothed_data

    def low_pass_filter(self, data, cutoff=0.1, fs=1.0):
        from scipy.signal import butter, filtfilt
        b, a = butter(4, cutoff, btype='low', fs=fs)
        batch, time, feature = data.shape
        smoothed_data = np.zeros_like(data)
        for b in range(batch):
            for f in range(feature):
                smoothed_data[b, :, f] = filtfilt(b, a, data[b, :, f])
        return smoothed_data

    def post_process(self, data):
        return data


# 示例使用
if __name__ == "__main__":

    seq_len = 96 * 6
    pred_len = 96
    dataset = get_dataset('Electricity')
    mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
        'test', 'OT', seq_len, Augmentor('none', False), 1000, 100
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

    denoise_methods = ['none', 'moving_average', 'ewma', 'median']

    fig, axs = plt.subplots(len(denoise_methods), 1, figsize=(12, 16))
    for i, method in enumerate(denoise_methods):
        denoiser = Denoiser(method=method)
        denoised_data = denoiser.pre_process(seqs.copy())
        axs[i].plot(seqs[0, :, 0], label='Original Data', color='blue')
        axs[i].plot(denoised_data[0, :, 0], label=f'Denoised Data ({method})', color='orange')
        axs[i].set_title(f'Denoised Data ({method})')
        axs[i].legend()
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Value')

    plt.tight_layout()
    plt.show()
