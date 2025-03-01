import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import DataLoader

from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import EttHour, CustomDataset, Electricity, Weather


class Normalizer:
    def __init__(self, method, mode, input_data=None, history_data=None, train_scaler=None):
        assert method in ['none', 'standard', 'minmax', 'maxabs', 'robust']
        assert mode in ['none', 'train', 'input', 'history']
        self.method = method
        self.mode = mode

        if mode == 'none' or method == 'none':
            return
        if mode == 'train':
            assert isinstance(train_scaler, (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler)), \
                'Invalid train scaler type: {}'.format(type(train_scaler))
            self.scaler = train_scaler
        elif mode in ['input', 'history']:
            data = input_data if mode == 'input' else history_data
            self.scaler_params = self._compute_scaler_params(data)
        else:
            raise Exception('Invalid normalizer mode: {}'.format(self.mode))

    def _compute_scaler_params(self, data):
        assert data.ndim == 3  # (batch, time, feature)
        batch, time, feature = data.shape
        params = {}
        for i in range(feature):
            feature_data = data[:, :, i].reshape(batch, -1)
            if self.method == 'standard':
                mean = np.mean(feature_data, axis=1, keepdims=True)
                std = np.std(feature_data, axis=1, keepdims=True)
                params[i] = (mean, std)
            elif self.method == 'minmax':
                min_val = np.min(feature_data, axis=1, keepdims=True)
                max_val = np.max(feature_data, axis=1, keepdims=True)
                params[i] = (min_val, max_val)
            elif self.method == 'maxabs':
                max_abs_val = np.max(np.abs(feature_data), axis=1, keepdims=True)
                params[i] = max_abs_val
            elif self.method == 'robust':
                median = np.median(feature_data, axis=1, keepdims=True)
                q1 = np.percentile(feature_data, 25, axis=1, keepdims=True)
                q3 = np.percentile(feature_data, 75, axis=1, keepdims=True)
                params[i] = (median, q1, q3)
        return params

    def pre_process(self, data):
        if self.mode == 'none' or self.method == 'none':
            return data
        assert data.ndim == 3  # (batch, time, feature)
        batch, time, feature = data.shape
        res = np.zeros_like(data)
        if self.mode == 'train':
            # 使用 sklearn 的 scaler 进行变换
            for i in range(feature):
                feature_data = data[:, :, i].reshape(-1, 1)
                res[:, :, i] = self.scaler.transform(feature_data).reshape(batch, time)
        else:
            # 使用自定义的 scaler 参数进行变换
            for i in range(feature):
                feature_data = data[:, :, i].reshape(batch, -1)
                if self.method == 'standard':
                    mean, std = self.scaler_params[i]
                    res[:, :, i] = ((feature_data - mean) / std).reshape(batch, time)
                elif self.method == 'minmax':
                    min_val, max_val = self.scaler_params[i]
                    res[:, :, i] = ((feature_data - min_val) / (max_val - min_val)).reshape(batch, time)
                elif self.method == 'maxabs':
                    max_abs_val = self.scaler_params[i]
                    res[:, :, i] = (feature_data / max_abs_val).reshape(batch, time)
                elif self.method == 'robust':
                    median, q1, q3 = self.scaler_params[i]
                    res[:, :, i] = ((feature_data - median) / (q3 - q1)).reshape(batch, time)
        return res

    def post_process(self, data):
        if self.mode == 'none' or self.method == 'none':
            return data
        assert data.ndim == 3  # (batch, time, feature)
        batch, time, feature = data.shape
        res = np.zeros_like(data)
        if self.mode == 'train':
            # 使用 sklearn 的 scaler 进行反变换
            for i in range(feature):
                feature_data = data[:, :, i].reshape(-1, 1)
                res[:, :, i] = self.scaler.inverse_transform(feature_data).reshape(batch, time)
        else:
            # 使用自定义的 scaler 参数进行反变换
            for i in range(feature):
                feature_data = data[:, :, i].reshape(batch, -1)
                if self.method == 'standard':
                    mean, std = self.scaler_params[i]
                    res[:, :, i] = (feature_data * std + mean).reshape(batch, time)
                elif self.method == 'minmax':
                    min_val, max_val = self.scaler_params[i]
                    res[:, :, i] = (feature_data * (max_val - min_val) + min_val).reshape(batch, time)
                elif self.method == 'maxabs':
                    max_abs_val = self.scaler_params[i]
                    res[:, :, i] = (feature_data * max_abs_val).reshape(batch, time)
                elif self.method == 'robust':
                    median, q1, q3 = self.scaler_params[i]
                    res[:, :, i] = (feature_data * (q3 - q1) + median).reshape(batch, time)
        return res


if __name__ == '__main__':

    # 示例调用
    history_seq = np.random.rand(3, 576, 1)  # 这是一个随机生成的示例数据
    normalizer = Normalizer('standard', 'input', history_seq, None, None)
    preprocessed_data = normalizer.pre_process(history_seq)
    postprocessed_data = normalizer.post_process(preprocessed_data)
    # print(preprocessed_data)
    # print(postprocessed_data)

    seq_len = 96 * 6
    pred_len = 96
    # dataset = EttHour(root_path='../_datasets/ts-data/ETT-small/', data_path='ETTh1.csv')
    # dataset = Electricity(root_path='../_datasets/ts-data/electricity/', data_path='electricity.csv')
    dataset = Weather(root_path='../_datasets/ts-data/weather/', data_path='weather.csv')
    mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
        'train', 'OT', seq_len, Augmentor('none', False), 4, 3
    custom_dataset = CustomDataset(dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
    idx, history, label = next(iter(dataloader))  # batch, time, feature
    print(f"len(dataloader): {len(dataloader)}")
    print(f"str(idx.tolist()): {str(idx.tolist())}")
    exit()
    scaler = dataset.get_train_scaler('standard', target_column)
    history = scaler.transform(history.numpy().reshape(-1, 1)).reshape(batch_size, seq_len, 1)
    label = scaler.transform(label.numpy().reshape(-1, 1)).reshape(batch_size, pred_len, 1)
    seqs = np.array(history)

    import matplotlib


    def is_pycharm():
        for key, value in os.environ.items():
            if key == "PYCHARM_HOSTED":
                print(f"PYCHARM_HOSTED={value}")
                return True


    matplotlib.use('TkAgg') if is_pycharm() else None

    normalizer_methods = ['none', 'standard', 'minmax', 'maxabs', 'robust']
    normalizer_modes = ['input', 'train']

    fig, axs = plt.subplots(len(normalizer_methods), len(normalizer_modes), figsize=(20, 20))
    for i, warper_method in enumerate(normalizer_methods):
        for j, warper_mode in enumerate(normalizer_modes):
            print(f'Processing Normalizer ({warper_method}, {warper_mode})')
            warper = Normalizer(warper_method, warper_mode, seqs, history_data=seqs, train_scaler=scaler)
            preprocessed_data = warper.pre_process(seqs)
            postprocessed_data = warper.post_process(preprocessed_data)
            axs[i, j].plot(preprocessed_data[0, :, 0], label=f'Pre-processed Data ({warper_method})', color='orange')
            axs[i, j].plot(postprocessed_data[0, :, 0], label=f'Post-processed Data ({warper_method})', color='red')
            axs[i, j].set_title(f'Method ({warper_method}), Mode: {warper_mode}')
            axs[i, j].legend()
    plt.show()
