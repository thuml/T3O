import logging
import os
from math import ceil

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from torch.utils.data import DataLoader

from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import get_dataset, CustomDataset



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
    augmentations = ['time_warp']

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
