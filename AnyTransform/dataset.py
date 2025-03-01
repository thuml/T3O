import logging
import os
from math import floor

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset

import itertools
from torch.utils.data import Sampler


class DynamicBatchSampler(Sampler):
    def __init__(self, data_source, batch_sizes):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_sizes = itertools.cycle(batch_sizes)
        assert len(batch_sizes) == 1 or np.sum(batch_sizes) == len(data_source), \
            f'Invalid batch_sizes: {batch_sizes}, data_source: {len(data_source)}'
        self.batch_indices = self._create_batch_indices()
        assert len(batch_sizes) == 1 or len(self.batch_indices) == len(batch_sizes), \
            f'Invalid batch_indices: {len(self.batch_indices)}, batch_sizes: {len(batch_sizes)}'

    def _create_batch_indices(self):
        indices = list(range(len(self.data_source)))
        batch_indices = []
        batch_size = next(self.batch_sizes)
        while indices:
            if len(indices) < batch_size:
                batch_size = len(indices)
            batch_indices.append(indices[:batch_size])
            indices = indices[batch_size:]
            batch_size = next(self.batch_sizes)
        return batch_indices

    def __iter__(self):
        return iter(self.batch_indices)

    def __len__(self):
        return len(self.batch_indices)


class CustomDataset(Dataset):
    def __init__(self, dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample):
        self.dataset = dataset
        self.mode = mode
        self.target_column = target_column
        self.max_seq_len = max_seq_len
        self.pred_len = pred_len
        self.augmentor = augmentor
        self.indices = self.dataset.get_available_idx_list(mode, max_seq_len, pred_len)
        if num_sample != 'all':
            selected_indices_indexes = np.linspace(0, len(self.indices) - 1, num_sample).astype(int)
            self.indices = [self.indices[i] for i in selected_indices_indexes]
        logging.info(f'{mode} dataset size: {len(self)}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        history_with_label = self.dataset.get_history_with_label(
            self.target_column, self.mode, real_idx, self.max_seq_len, self.pred_len)
        # 进行数据增强
        _history_with_label = history_with_label.reshape((1, -1, 1))  # batch time feature
        aug_method = 'none'
        if self.mode != 'test':
            aug_method = self.augmentor.get_aug_method()
            _history_with_label = self.augmentor.apply_augmentation(_history_with_label)
        # ！！！: shape可能会变成 (len(aug_methods), self.max_seq_len + self.pred_len, 1) # aug_batch time feature
        assert _history_with_label.shape[2] == 1 and _history_with_label.shape[1] == self.max_seq_len + self.pred_len, \
            f'Invalid history_with_label shape: {_history_with_label.shape}'
        history, label = _history_with_label[:, :-self.pred_len, :], _history_with_label[:, -self.pred_len:, :]
        return real_idx, aug_method, history, label

class CustomDatasetCov(Dataset):
    def __init__(self, dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample):
        self.dataset = dataset
        self.mode = mode
        self.target_column = target_column
        self.max_seq_len = max_seq_len
        self.pred_len = pred_len
        self.augmentor = augmentor
        self.indices = self.dataset.get_available_idx_list(mode, max_seq_len, pred_len)
        if num_sample != 'all':
            selected_indices_indexes = np.linspace(0, len(self.indices) - 1, num_sample).astype(int)
            self.indices = [self.indices[i] for i in selected_indices_indexes]
        logging.info(f'{mode} dataset size: {len(self)}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        history_with_label, cov_history_with_label = self.dataset.get_cov_history_with_label(
            self.target_column, self.mode, real_idx, self.max_seq_len, self.pred_len)
        # 进行数据增强
        _history_with_label = history_with_label.reshape((1, -1, 1))  # batch time feature
        S,C = cov_history_with_label.shape
        _cov_history_with_label = cov_history_with_label.reshape((1, S, C))
        aug_method = 'none'
        if self.mode != 'test':
            aug_method = self.augmentor.get_aug_method()
            _history_with_label = self.augmentor.apply_augmentation(_history_with_label)
        # ！！！: shape可能会变成 (len(aug_methods), self.max_seq_len + self.pred_len, 1) # aug_batch time feature
        assert _history_with_label.shape[2] == 1 and _history_with_label.shape[1] == self.max_seq_len + self.pred_len, \
            f'Invalid history_with_label shape: {_history_with_label.shape}'
        history, label = _history_with_label[:, :-self.pred_len, :], _history_with_label[:, -self.pred_len:, :]
        cov_history, cov_label = _cov_history_with_label[:, :-self.pred_len, :], _cov_history_with_label[:, -self.pred_len:, :]
        return real_idx, aug_method, history, label, cov_history, cov_label # 重载出其他变量；增广只在 OT 上进行

def split721(train_len, val_len, test_len):
    total_len = train_len + val_len + test_len
    ratios = [0.7, 0.2, 0.1]
    train_len = floor(total_len * ratios[0])
    val_len = floor(total_len * ratios[1])
    test_len = total_len - train_len - val_len
    return train_len, val_len, test_len


def get_dataset(data_name, fast_split=None):
    fast_split = fast_split if fast_split is not None else False  # FIXME: 让test样本多些公平些 但是Uni2ts太慢了

    if data_name == 'ETTh1':
        dataset = EttHour('../DATA/ETT-small', 'ETTh1.csv', fast_split)
    elif data_name == 'ETTh2':
        dataset = EttHour('../DATA/ETT-small/', 'ETTh2.csv', fast_split)
    elif data_name == 'ETTm1':
        dataset = EttMinute('../DATA/ETT-small/', 'ETTm1.csv', fast_split)
    elif data_name == 'ETTm2':
        dataset = EttMinute('../DATA/ETT-small/', 'ETTm2.csv', fast_split)
    elif data_name == 'Exchange' or data_name == 'exchange_rate':
        dataset = Exchange('../DATA/exchange_rate/', 'exchange_rate.csv', fast_split)
    elif data_name == 'Weather' or data_name == 'weather':
        dataset = Weather('../DATA/weather/', 'weather.csv', fast_split)
    elif data_name == 'Electricity' or data_name == 'electricity':
        dataset = Electricity('../DATA/electricity/', 'electricity.csv', fast_split)
    elif data_name == 'Traffic' or data_name == 'traffic':
        dataset = Traffic('../DATA/traffic/', 'traffic.csv', fast_split)
    else:
        raise ValueError(f"Unknown data_name: {data_name}")
    return dataset


class MyDataBase:
    column_names = ['OT']
    # 分开定义目标变量与其他变量
    def __init__(self, root_path, data_path, train_start, train_end, val_start, val_end, test_start, test_end):
        self.root_path = root_path
        self.data_path = data_path
        self.train_start, self.train_end = train_start, train_end
        self.val_start, self.val_end = val_start, val_end
        self.test_start, self.test_end = test_start, test_end
        self.train_len = self.train_end - self.train_start
        self.val_len = self.val_end - self.val_start
        self.test_len = self.test_end - self.test_start
        self.total_len = self.train_len + self.val_len + self.test_len

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # 丢掉第一列date
        df_raw = df_raw.iloc[:, 1:]
        self.df_data = df_raw
        # dict: column:np.1dArray
        self.np_data_dict = {col: np.array(df_raw[col].values).reshape(-1) for col in df_raw.columns}

        self.scalers = {}
        # self.train_statistics = {}

    def get_history_with_label(self, target, flag, real_idx, max_seq_len, pred_len):
        # flag: train, test, val
        # target: column:   HUFL	HULL	MUFL	MULL	LUFL	LULL	OT
        assert flag in ['train', 'val', 'test'], \
            f'Invalid flag: {flag}'
        assert target in self.column_names, \
            f'Invalid target: {target}'
        # real_idx = idx + self.__getattribute__(flag + '_start')
        assert real_idx - max_seq_len >= 0, \
            f'Invalid real_idx: {real_idx}, max_seq_len: {max_seq_len}'
        assert real_idx + pred_len < self.__getattribute__(flag + '_end'), \
            f'Invalid real_idx: {real_idx}, pred_len: {pred_len}'
        # history = self.np_data_dict[target][real_idx - max_seq_len: real_idx]
        # label = self.np_data_dict[target][real_idx: real_idx + pred_len]
        history_with_label = self.np_data_dict[target][real_idx - max_seq_len: real_idx + pred_len]
        
        return history_with_label
    
    def get_cov_history_with_label(self, target, flag, real_idx, max_seq_len, pred_len):
        # flag: train, test, val
        # target: column:   HUFL	HULL	MUFL	MULL	LUFL	LULL	OT
        assert flag in ['train', 'val', 'test'], \
            f'Invalid flag: {flag}'
        assert target in self.column_names, \
            f'Invalid target: {target}'
        # real_idx = idx + self.__getattribute__(flag + '_start')
        assert real_idx - max_seq_len >= 0, \
            f'Invalid real_idx: {real_idx}, max_seq_len: {max_seq_len}'
        assert real_idx + pred_len < self.__getattribute__(flag + '_end'), \
            f'Invalid real_idx: {real_idx}, pred_len: {pred_len}'
        # history = self.np_data_dict[target][real_idx - max_seq_len: real_idx]
        # label = self.np_data_dict[target][real_idx: real_idx + pred_len]
        history_with_label = self.np_data_dict[target][real_idx - max_seq_len: real_idx + pred_len]
        cov_history_with_label = self.df_data.values[real_idx - max_seq_len: real_idx + pred_len, :-1]
        return history_with_label, cov_history_with_label

    def get_available_idx_list(self, mode, max_seq_len, pred_len):
        # max_seq_len 一般取train_len/2
        # max_pred_len 一般取Patch=96
        assert mode in ['train', 'val', 'test'], \
            f'Invalid flag: {mode}'
        assert max_seq_len <= self.train_len / 2, \
            f'Invalid max_seq_len: {max_seq_len}, train_len: {self.train_len}'
        if mode == 'train':
            start = self.train_start + max_seq_len
            end = self.train_end - pred_len
        elif mode == 'val':
            start = self.val_start
            end = self.val_end - pred_len
        else:
            start = self.test_start
            end = self.test_end - pred_len
        assert start < end, \
            f'Invalid start: {start}, end: {end}'
        # real split idx in raw dataset!
        return list(range(start, end))

    def get_mode_scaler(self, mode, method, target):
        assert target in self.column_names
        assert mode in ['train', 'val', 'test']
        # assert method in ['none', 'standard', 'minmax', 'maxabs', 'robust']
        assert method in ['none', 'standard', 'robust']

        if method == 'none':
            return None

        if (mode, target, method) in self.scalers:
            return self.scalers[(mode, target, method)]

        # Create and fit new scaler only if not already computed
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError('Invalid scaler method')
        # Fit scaler to training data for the specified target
        if mode == 'train':
            mode_data = self.np_data_dict[target][self.train_start:self.train_end].reshape(-1, 1)
        elif mode == 'val':
            mode_data = self.np_data_dict[target][self.val_start:self.val_end].reshape(-1, 1)
        elif mode == 'test':
            mode_data = self.np_data_dict[target][self.test_start:self.test_end].reshape(-1, 1)
        else:
            raise ValueError('Invalid mode')
        scaler.fit(mode_data)#! Is fitting scaler on validate and test data applicable?
        # Store the scaler for future use
        self.scalers[(mode, target, method)] = scaler
        return scaler
    
    def get_scaler(self, method, target):
        assert target in self.column_names
        assert method in ['none', 'standard', 'robust']

        if method == 'none':
            return None

        if (target, method) in self.scalers:
            return self.scalers[(target, method)]

        # Create and fit new scaler only if not already computed
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError('Invalid scaler method')
        # Fit scaler to training data
        mode_data = self.np_data_dict[target][self.train_start:self.train_end].reshape(-1, 1)
        scaler.fit(mode_data)#! Is fitting scaler on validate and test data applicable?
        # Store the scaler for future use
        self.scalers[(target, method)] = scaler
        return scaler

    # def get_train_statistics_dict(self, target):  # 不被inputer使用，因为害怕数据偏移，用max_seq
    #     assert target in self.column_names
    #     if target in self.train_statistics:
    #         return self.train_statistics[target]
    #     train_data = self.np_data_dict[target][self.train_start:self.train_end]
    #     self.train_statistics[target] = {
    #         'mean': np.mean(train_data),
    #         'std': np.std(train_data),
    #         'min': np.min(train_data),
    #         'max': np.max(train_data),
    #         'median': np.median(train_data),
    #         'upper_quartile': np.percentile(train_data, 75),
    #         'lower_quartile': np.percentile(train_data, 25)
    #     }
    #     return self.train_statistics[target]


class EttMinute(MyDataBase):
    # column_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    column_names = ['OT']

    def __init__(self, root_path, data_path, fast_test_split):
        train_len, val_len, test_len = 34465, 11521, 11521
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 0, train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(EttMinute, self).__init__(root_path, data_path,
                                        train_start, train_end,
                                        val_start, val_end,
                                        test_start, test_end)


class EttHour(MyDataBase):
    # column_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    column_names = ['OT']

    def __init__(self, root_path, data_path, fast_test_split):
        # ETTh (8545, 2881, 2881) # 12 * 30 * 24=8640 有点差别！
        train_len, val_len, test_len = 8545, 2881, 2881
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 0, train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(EttHour, self).__init__(root_path, data_path,
                                      train_start, train_end,
                                      val_start, val_end,
                                      test_start, test_end)


class Exchange(MyDataBase):
    # column_names = ['0', '1', '2', '3', '4', '5', '6', 'OT']
    column_names = ['OT']

    def __init__(self, root_path, data_path, fast_test_split):
        # train_len, val_len, test_len = 5120, 665, 1442 # val太少了不够720
        train_len, val_len, test_len = 4343, 1442, 1442
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 0, train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(Exchange, self).__init__(root_path, data_path,
                                       train_start, train_end,
                                       val_start, val_end,
                                       test_start, test_end)


class Weather(MyDataBase):
    column_names = ['OT']

    def __init__(self, root_path, data_path, fast_test_split, clean=False):
        # clean=True 是因为影响到train_scaler的std进而影响test_mse的计算了。？考虑计算mse时不scale
        train_len, val_len, test_len = 36792, 5271, 10540
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 0, train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(Weather, self).__init__(root_path, data_path,
                                      train_start, train_end,
                                      val_start, val_end,
                                      test_start, test_end)
        # if clean:
        #     # OT数据的train部分存在两段连续的-9999的异常值，用moving_average的方式填充，窗口为1000，填充值从前往后
        #     train_idxes = np.arange(self.train_start, self.train_end)
        #     anomaly_idxes = np.where(self.np_data_dict['OT'][train_idxes] == -9999)[0]
        #     print(f'Anomaly count: {np.sum(anomaly_idxes)}')
        #     print(f'Anomaly idxes: {anomaly_idxes}')
        #     for idx in anomaly_idxes:
        #         fill_value = np.mean(self.np_data_dict['OT'][idx - 1000:idx])
        #         self.np_data_dict['OT'][idx] = fill_value


class Traffic(MyDataBase):
    column_names = ['OT']

    def __init__(self, root_path, data_path, fast_test_split):
        train_len, val_len, test_len = 12185, 1757, 3509
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 0, train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(Traffic, self).__init__(root_path, data_path,
                                      train_start, train_end,
                                      val_start, val_end,
                                      test_start, test_end)


class Electricity(MyDataBase):
    # 0  1	2	3	4	5	6	OT
    # column_names = ['0', '1', '2', '3', '4', '5', '6', 'OT']
    column_names = ['OT']

    def __init__(self, root_path, data_path, fast_test_split):
        # Exchange (1000, 1000, 1000)
        train_len, val_len, test_len = 18317, 2633, 5261
        if fast_test_split:
            train_len, val_len, test_len = split721(train_len, val_len, test_len)

        train_start, train_end = 0, train_len
        val_start, val_end = train_len, train_len + val_len
        test_start, test_end = train_len + val_len, train_len + val_len + test_len
        super(Electricity, self).__init__(root_path, data_path,
                                          train_start, train_end,
                                          val_start, val_end,
                                          test_start, test_end)


if __name__ == '__main__':
    # dataset = get_dataset('Electricity', True)
    # dataset = get_dataset('Weather', True)
    # dataset = get_dataset('ETTm2', True)
    dataset = get_dataset('Traffic', True)
    # 画个图吧
    import matplotlib
    import matplotlib.pyplot as plt


    def is_pycharm():
        for key, value in os.environ.items():
            if key == "PYCHARM_HOSTED":
                print(f"PYCHARM_HOSTED={value}")
                return True


    matplotlib.use('TkAgg') if is_pycharm() else None

    fig, axs = plt.subplots(4, 1, figsize=(20, 10))
    axs[0].plot(dataset.np_data_dict['OT'])
    axs[1].plot(dataset.np_data_dict['OT'][dataset.train_start:dataset.train_end])
    axs[2].plot(dataset.np_data_dict['OT'][dataset.val_start:dataset.val_end])
    axs[3].plot(dataset.np_data_dict['OT'][dataset.test_start:dataset.test_end])
    plt.show()
    # 一般来说，我们可以取train_len/2作为max_seq_len
    # 一般来说，我们可以取96作为max_pred_len
    # max_seq_len = 12 * 30 * 24
    # max_pred_len = 96
    # for idx in dataset.get_available_idx_list('train', max_seq_len, max_pred_len):
    #     max_seq, label = dataset.get_max_seq_and_label('OT', 'train', idx, max_seq_len, max_pred_len)
    #     print(max_seq.shape, label.shape)
    #     break
    # scaler = dataset.get_scaler('standard', 'OT')
    # max_seq = scaler.transform(max_seq.reshape(-1, 1)).reshape(-1)
    # label = scaler.transform(label.reshape(-1, 1)).reshape(-1)
    # print(max_seq.shape, label.shape)
    # max_seq = scaler.inverse_transform(max_seq.reshape(-1, 1)).reshape(-1)
    # label = scaler.inverse_transform(label.reshape(-1, 1)).reshape(-1)
    # print(max_seq.shape, label.shape)
    # print(max_seq, label)
    # pass
