import os
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader

from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import get_dataset, CustomDataset


def time_start():
    return time.time()


def log_time_delta(t, event_name):
    d = time.time() - t
    print(f"{event_name} time: {d}")


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
        else:
            raise ValueError(f"Unsupported denoise method: {self.method}")

    def moving_average(self, data):
        window_size = self.window_size
        kernel = np.ones(window_size) / window_size
        smoothed_data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=data)
        return smoothed_data

    def ewma(self, data):
        alpha = self.alpha
        batch, time, feature = data.shape
        smoothed_data = np.zeros_like(data)
        smoothed_data[:, 0, :] = data[:, 0, :]  # Initialize with the first value

        for t in range(1, time):
            smoothed_data[:, t, :] = alpha * data[:, t, :] + (1 - alpha) * smoothed_data[:, t - 1, :]
        return smoothed_data

    def median_filter(self, data):
        window_size = self.window_size
        pad_size = window_size // 2
        padded_data = np.pad(data, ((0, 0), (pad_size, pad_size), (0, 0)), mode='edge')
        smoothed_data = np.apply_along_axis(lambda m: self._apply_median_filter(m, window_size), axis=1, arr=padded_data)
        return smoothed_data[:, pad_size:-pad_size, :]

    def _apply_median_filter(self, data, window_size): # 时间有点长
        return np.array([np.median(data[i:i + window_size]) for i in range(len(data) - window_size + 1)])

    def low_pass_filter(self, data, cutoff=0.1, fs=1.0):
        b, a = butter(4, cutoff, btype='low', fs=fs)
        smoothed_data = np.apply_along_axis(lambda m: filtfilt(b, a, m), axis=1, arr=data)
        return smoothed_data

    def post_process(self, data):
        return data

# 示例使用
if __name__ == "__main__":

    seq_len = 96 * 6
    pred_len = 192
    dataset = get_dataset('Electricity')
    mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
        'test', 'OT', seq_len, Augmentor('none', False), 6000, 600
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
        t = time_start()
        denoised_data = denoiser.pre_process(seqs.copy())
        # Pre-process (none) time: 0.0005359649658203125
        # Pre-process (moving_average) time: 0.002805948257446289
        # Pre-process (ewma) time: 0.003202676773071289
        # Pre-process (median) time: 1.9631881713867188
        log_time_delta(t, f"Pre-process ({method})")
        axs[i].plot(seqs[0, :, 0], label='Original Data', color='blue')
        axs[i].plot(denoised_data[0, :, 0], label=f'Denoised Data ({method})', color='orange')
        axs[i].set_title(f'Denoised Data ({method})')
        axs[i].legend()
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Value')

    plt.tight_layout()
    plt.show()
