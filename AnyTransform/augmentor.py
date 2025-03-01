import logging
import os
from math import ceil

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from torch.utils.data import DataLoader

from AnyTransform.dataset import get_dataset, CustomDataset


def moving_average_smooth(data):
    window_size = 3
    kernel = np.ones(window_size) / window_size
    smoothed_data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=data)
    return smoothed_data


def ewma_smooth(x):
    alpha = 0.3
    batch, time, feature = x.shape
    smoothed_data = np.zeros_like(x)
    smoothed_data[:, 0, :] = x[:, 0, :]  # Initialize with the first value

    for t in range(1, time):
        smoothed_data[:, t, :] = alpha * x[:, t, :] + (1 - alpha) * smoothed_data[:, t - 1, :]
    return smoothed_data


def fft_denoise(data, percentile=90):
    batch, time, feature = data.shape
    denoised_data = np.zeros_like(data)

    for b in range(batch):
        for f in range(feature):
            # Perform FFT
            fft_coeffs = np.fft.fft(data[b, :, f])
            # Get magnitudes and set coefficients below threshold to zero
            magnitudes = np.abs(fft_coeffs)
            upper_magnitude = np.percentile(magnitudes, percentile)
            fft_coeffs[magnitudes < upper_magnitude] = 0 + 0j
            # Perform inverse FFT
            denoised_data[b, :, f] = np.fft.ifft(fft_coeffs).real
    return denoised_data


def jitter(x): # 感觉有点太狠了
    factor = 0.03
    x_new = np.zeros(x.shape)
    for i in range(x.shape[0]):
        range_values = np.max(x[i], axis=0) - np.min(x[i], axis=0)
        for j in range(x.shape[2]):
            x_new[i, :, j] = x[i, :, j] + np.random.normal(loc=0., scale=range_values[j] * factor, size=x.shape[1])
    return x_new


def outlier(x, pred_len=None, factor=2):
    x_new = x.copy()
    max_values = np.max(x, axis=1, keepdims=True)
    min_values = np.min(x, axis=1, keepdims=True)
    range_values = max_values - min_values
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            time_len = x.shape[1]
            # time_idx = time_len - pred_len - 1

            time_idx = np.random.randint(time_len * 0.6, time_len * 0.99)
            time_idx = min(time_len - pred_len - 1, time_idx)  # 让outlier仅在seq里

            # time_idx = np.random.randint(time_len * 0.6, time_len * 0.9)
            x_new[i, time_idx, j] = x[i, time_idx, j] - range_values[i, 0, j] * factor
    return x_new


def scaling(x, factor=1.5):
    return x * factor


def window_slice(x, reduce_ratio=0.9):
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len),
                                       pat[starts[i]:ends[i], dim]).T
    return ret


def time_warp(x, sigma=0.02, knot=4):
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return ret


def magnitude_warp(x, sigma=0.2, knot=4):  # our:10
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array(
            [CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret


def vertical_translation(x):
    factor = np.random.choice([-2, 2])  # FIXME: 保证每个params和每个split_idx计算mse的公平！！！
    # factor = np.random.choice([-0.25, 0.25])  # FIXME: 保证每个params和每个split_idx计算mse的公平！！！
    mean = np.mean(x, axis=1, keepdims=True)
    return x + mean * factor


def horizontal_translation(x, factor=None):
    factor = 1 / 5 if factor is not None else factor
    shift = ceil(x.shape[1] * factor)
    return np.roll(x, shift, axis=1)  # FIXME: 考虑用pad？


# def translation_first(x, factor=1):
#     shift = x[0] * factor
#     return x + shift


def scale_up_around_mean(x):
    # factor = np.random.choice([1 / 10, -1 / 10, 10, -10]) # 保证每个params样本计算mse的公平！！！
    factor = np.random.choice([2, -2])
    mean = np.mean(x, axis=1, keepdims=True)
    return mean + (x - mean) * factor


def scale_down_around_mean(x):
    factor = np.random.choice([1 / 2, -1 / 2])
    mean = np.mean(x, axis=1, keepdims=True)
    return mean + (x - mean) * factor


def scaling_around_first(x, factor=2):
    # x0 = x[0]
    # return x0 + (x - x0) * factor

    factor = np.random.choice([0.6])
    # 数据维度（batch，time，feature）
    mean = np.mean(x, axis=1, keepdims=True)
    return mean + (x - mean) * factor

def slope_around_mean(x, angle=None):
    # Determine the angle if not provided
    angle = np.random.choice([15, -15]) if angle is None else angle  # 60才明显一点点... # 不需要明显，需要接近真实
    # Convert angle to radians and calculate the slope
    radians = np.deg2rad(angle)
    slope = np.tan(radians)
    # Generate time indices
    time = np.arange(x.shape[1]).reshape(1, -1, 1)
    # Calculate the data range
    data_range = np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True)
    # Normalize the slope to the data range
    normalized_slope = slope * (data_range / x.shape[1])
    # Create the slope effect to add to the data
    added = (time * normalized_slope).reshape(1, -1, 1)
    # Center the added slope effect around the mean
    added = added - np.mean(added, axis=1, keepdims=True)
    # Print the added slope effect for debugging
    # Add the slope effect to the original data
    return x + added


def slope_at_split(x, angle=None):
    # Determine the angle if not provided
    angle = np.random.choice([60, -60]) if angle is None else angle  # 60才明显一点点... # 不需要明显，需要接近真实
    split_ratio = np.random.uniform(0.7, 0.9)
    # Convert angle to radians and calculate the slope
    radians = np.deg2rad(angle)
    slope = np.tan(radians)
    # Generate time indices
    time = np.arange(x.shape[1]).reshape(1, -1, 1)
    # Calculate the data range
    data_range = np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True)
    # Normalize the slope to the data range
    normalized_slope = slope * (data_range / x.shape[1])
    # Create the slope effect to add to the data
    added = (time * normalized_slope).reshape(1, -1, 1)
    # Center the added slope effect around the mean
    added[:, :int(split_ratio * x.shape[1]), :] = 0
    # Add the slope effect to the original data

    return x + added


def turn_around_mean(x):
    angle = np.random.choice([100, -100])
    x1 = x
    x2 = slope_around_mean(x1, angle)
    split = np.random.randint(x.shape[1] * 0.7, x.shape[1] * 0.9)
    gap = x2[:, split:split + 1, :] - x1[:, split - 1:split, :]
    x3 = x2 - gap
    return np.concatenate([x1[:, :split], x3[:, split:]], axis=1)


def sin_around_mean(x, amplitude=None, frequency=None):
    # Determine the amplitude and frequency if not provided
    amplitude = np.random.uniform(0.5, 1.5) if amplitude is None else amplitude  # Default amplitude range
    frequency = np.random.uniform(0.1, 1.0) if frequency is None else frequency  # Default frequency range

    # Generate time indices
    time = np.arange(x.shape[1]).reshape(1, -1, 1)

    # Calculate the data range
    data_range = np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True)

    # Normalize the amplitude to the data range
    normalized_amplitude = amplitude * (data_range / 2)

    # Create the sine effect to add to the data
    added = normalized_amplitude * np.sin(2 * np.pi * frequency * time / x.shape[1])

    # Center the added sine effect around the mean
    added = added - np.mean(added, axis=1, keepdims=True)

    # Add the sine effect to the original data
    return x + added


def magnitude_flip(x):
    mean = np.mean(x, axis=1, keepdims=True)
    residual = x - mean
    return mean - residual


class Augmentor:
    def __init__(self, aug_method, mode, pred_len):
        # 高质量的warp
        # FIXME:重要的是要向着test出发
        self.org_aug_method_dict = {  # 越靠前越重要
            # 'none1': lambda x: x,
            # 'none2': lambda x: x,
            # 'none3': lambda x: x,
            # 'none4': lambda x: x,
            # 'none5': lambda x: x,
            # 'none6': lambda x: x,
            # 'none7': lambda x: x,
            # 'none8': lambda x: x,
            # 'none9': lambda x: x,
            'none': lambda x: x,
            'magnitude_flip': magnitude_flip,
            'time_flip': lambda x: x[:, ::-1, :].copy(),  # 稳定产生新样本

            'window_slice': window_slice,
            'time_warp': time_warp,
            'magnitude_warp': magnitude_warp,
            # 15*96 // 2*96 = 7.5
            # 'horizontal_translation': horizontal_translation,  # 偏移大 产生高质量数据！！！(不如新sample？
            'horizontal_translation1': lambda x: horizontal_translation(x, 1 / 5),
            # 'horizontal_translation2': lambda x: horizontal_translation(x, 2 / 5),
            # 'horizontal_translation3': lambda x: horizontal_translation(x, 3 / 5),
            # 'horizontal_translation4': lambda x: horizontal_translation(x, 1 / 4),
            # 'horizontal_translation5': lambda x: horizontal_translation(x, 2 / 4),
            'ewma_smooth': ewma_smooth,
            'jitter': jitter,
            # 保证val下的每个params都被对应pred_len的outlier恶心到！！！
            'outlier': lambda x: outlier(x, pred_len, 2),
            'outlier2': lambda x: outlier(x, pred_len, 1),
            # 'outlier3': lambda x: outlier(x, pred_len),
            # 'outlier': lambda x: outlier(x, 192),  # 多个outlier很重要。。。。
            # 'outlier2': lambda x: outlier(x, 96),
            # 'outlier3': lambda x: outlier(x, 48),
            # 'outlier4': lambda x: outlier(x, 24),
            # 'fft_denoise': fft_denoise,

            # 'slope_around_mean': slope_around_mean,  # 不如magnitude_warp -> Weather存在巨陡峭
            'turn_around_mean': turn_around_mean,  # 大幅度对Weather很重要！！！
            # 'sin_around_mean': sin_around_mean, # 跟magnitude_warp一样
            # 'moving_average_smooth': moving_average_smooth, # 不明显变化，重复？
            # 'vertical_translation': vertical_translation,  # 可能model普遍内置mean导致无效 # 可能纯mean改变不好
            # 'scale_down_around_mean': scale_down_around_mean, # 有scaleUp了 # 不如magnitude_warp
            # 'scale_up_around_mean': scale_up_around_mean,  # 不如magnitude_warp
            # 'scaling': scaling, # 不如magnitude_warp
            # 'translation_first': translation_first, # 重复
            'scaling_around_first': scaling_around_first, # 不如magnitude_warp # 重复

            # FIXME: 纯粹画图
            # 'vertical_translation': vertical_translation,  # 可能model普遍内置mean导致无效 # 可能纯mean改变不好

        }
        self.mode = mode
        assert mode in ['fix', 'rotate', 'all']
        self.aug_method_dict = None
        if mode == 'fix':
            self.aug_method_dict = {aug_method: self.org_aug_method_dict[aug_method]}
        elif mode == 'rotate':  # rotate模式时保证none的比例占一半
            self.aug_method_dict = self.org_aug_method_dict.copy()
            aug_method_len = len(self.org_aug_method_dict.keys())
            for i in range(aug_method_len):
                self.aug_method_dict[f'none{i}'] = lambda x: x
        elif mode == 'all':
            self.aug_method_dict = self.org_aug_method_dict.copy()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.aug_idx = None
        self.reset_aug_method(aug_method)

    def reset_aug_method(self, aug_method):
        assert aug_method in self.aug_method_dict.keys(), f"Unknown augmentation method: {aug_method}"
        self.aug_idx = list(self.aug_method_dict.keys()).index(aug_method)
        logging.info(f"Augmentation method reset to {aug_method}")

    def get_aug_method(self):
        return list(self.aug_method_dict.keys())[self.aug_idx]

    def apply_augmentation(self, data):
        assert len(data.shape) == 3 and data.shape[0] == 1 and data.shape[2] == 1, \
            f"Data shape must be (1, time, 1), got {data.shape}"
        old_shape = data.shape
        if self.mode == 'fix':
            aug_method_name = list(self.aug_method_dict.keys())[self.aug_idx]
            augmented_data = self.aug_method_dict[aug_method_name](data)
            assert augmented_data.shape == old_shape, \
                f"Data shape changed from {old_shape} to {augmented_data.shape}, aug_method_name: {aug_method_name}"
        elif self.mode == 'rotate':
            aug_method_name = list(self.aug_method_dict.keys())[self.aug_idx]
            augmented_data = self.aug_method_dict[aug_method_name](data)
            self.aug_idx = (self.aug_idx + 1) % len(self.aug_method_dict)
            assert augmented_data.shape == old_shape, \
                f"Data shape changed from {old_shape} to {augmented_data.shape}, aug_method_name: {aug_method_name}"
        elif self.mode == 'all':
            # shape[0] 会增加
            augmented_data = np.zeros((len(self.aug_method_dict), data.shape[1], data.shape[2]))
            for i, aug_method in enumerate(self.aug_method_dict.keys()):
                augmented_data[i] = self.aug_method_dict[aug_method](data)
            assert augmented_data.shape == (len(self.aug_method_dict), old_shape[1], old_shape[2]), \
                f"Data shape changed from {old_shape} to {augmented_data.shape}"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return augmented_data


# Example usage with plotting
if __name__ == "__main__":
    # ETTm1 49986

    seq_len = 96 * 4
    pred_len = 192
    # dataset = get_dataset('Exchange')
    # dataset = get_dataset('Electricity')
    # dataset = get_dataset('Electricity')
    dataset = get_dataset('ETTm1')
    # dataset = get_dataset('Weather')
    # dataset = get_dataset('ETTh1')
    mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
        'test', 'OT', seq_len, Augmentor('none', 'fix', pred_len), 10000, 1000
    custom_dataset = CustomDataset(dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
    iter = iter(dataloader)
    while True:
        idxes, aug_methods, history, label = next(iter)
        print(idxes)
        if 49986 in list(idxes):
            print("found")
            idxes_idx = list(idxes).index(49986)
            data = history.reshape(-1, seq_len, 1).numpy()[idxes_idx:idxes_idx + 1]
            break

    import matplotlib


    def is_pycharm():
        for key, value in os.environ.items():
            if key == "PYCHARM_HOSTED":
                print(f"PYCHARM_HOSTED={value}")
                return True


    matplotlib.use('TkAgg') if is_pycharm() else None

    # Augmentations to apply
    # augmentations = ['scaling', 'window_slice', 'time_warp', 'magnitude_warp', 'translation', 'scaling_around_mean',
    #                  'scaling_around_first', 'translation_first']
    # augmentations = Augmentor('none', 'all', pred_len).aug_method_dict.keys()
    # augmentations = ['fft_denoise', 'ewma_smooth']
    augmentations = ['scaling_around_first']

    # Create subplots
    fig, axes = plt.subplots(len(augmentations), 1, figsize=(ceil((seq_len + pred_len) / 96) * 5, 5))

    # Handle the case when there's only one subplot
    if len(augmentations) == 1:
        axes = [axes]

    for ax, aug_method in zip(axes, augmentations):
        print()
        augmentor = Augmentor(aug_method, 'fix', pred_len)
        augmented_data = augmentor.apply_augmentation(data)
        ax.plot(data[0, :, 0], label='Original', color='orange')
        if aug_method != 'none':
            ax.plot(augmented_data[0, :, 0], label='Augmented', color='blue')
        ax.legend()
        ax.set_title(aug_method)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

    plt.tight_layout()
    plt.show()
