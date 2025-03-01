import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ifft
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import Weather, CustomDataset
from AnyTransform.model import Uni2ts


class Decomposer:
    def __init__(self, period, component_for_model):
        self.period = period
        self.component_for_model = component_for_model.split('+')
        self.all_component_flag = component_for_model == 'all'
        self.trend = None
        self.season = None
        self.residual = None

    def pre_process(self, data):
        if target == 'all':
            return data

        trends, seasons, residuals = [], [], []

        for i in range(data.shape[0]):
            series = data[i, :, 0]  # Extract the series for each batch
            stl = STL(series, period=self.period, seasonal=13)
            result = stl.fit()

            trends.append(result.trend)
            seasons.append(result.seasonal)
            residuals.append(result.resid)

        self.trend = np.array(trends).reshape(data.shape)
        self.season = np.array(seasons).reshape(data.shape)
        self.residual = np.array(residuals).reshape(data.shape)

        # Determine which components to return based on the target
        combined = np.zeros_like(data)
        if 'trend' in self.component_for_model:
            combined += self.trend
        if 'season' in self.component_for_model:
            combined += self.season
        if 'residual' in self.component_for_model:
            combined += self.residual

        return combined

    def post_process(self, pred):
        if self.all_component_flag:
            return pred
        pred_trend, pred_season, pred_residual = 0, 0, 0

        if 'trend' not in self.component_for_model:
            pred_trend = self._predict_trend(pred.shape[1])

        if 'season' not in self.component_for_model:
            pred_season = self._predict_season(pred.shape[1])

        if 'residual' not in self.component_for_model:
            pred_residual = self._predict_residual(pred.shape[1])

        return pred + pred_trend + pred_season + pred_residual

    def _predict_trend(self, pred_len):  # 用线性回归预测
        pred_trend = np.zeros((self.trend.shape[0], pred_len, 1))
        for i in range(self.trend.shape[0]):
            y = self.trend[i, :, 0]
            x = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            future_x = np.arange(len(y), len(y) + pred_len).reshape(-1, 1)
            pred_trend[i, :, 0] = model.predict(future_x)
        return pred_trend

    # def _predict_season(self, pred_len):  # 复制最近的单周期数据
    #     pred_season = np.zeros((self.season.shape[0], pred_len, 1))
    #     for i in range(self.season.shape[0]):
    #         y = self.season[i, :, 0]
    #         season_length = self.period
    #         pred_season[i, :, 0] = np.tile(y[-season_length:], int(np.ceil(pred_len / season_length)))[:pred_len]
    #     return pred_season

    def _predict_season(self, pred_len):
        pred_season = np.zeros((self.season.shape[0], pred_len, 1))
        for i in range(self.season.shape[0]):
            y = self.season[i, :, 0]
            n = len(y)
            f = fft(y)
            frequencies = np.fft.fftfreq(n)
            # 只保留重要的频率
            threshold = 0.1
            f[np.abs(frequencies) > threshold] = 0
            future_frequencies = np.fft.fftfreq(n + pred_len)
            future_f = np.zeros_like(future_frequencies, dtype=complex)
            future_f[:n] = f
            future_f = ifft(future_f).real
            pred_season[i, :, 0] = future_f[-pred_len:]
        return pred_season

    # def _predict_residual(self, pred_len):  # 滑动平均做预测值
    #     pred_residual = np.zeros((self.residual.shape[0], pred_len, 1))
    #     for i in range(self.residual.shape[0]):
    #         y = self.residual[i, :, 0]
    #         pred_residual[i, :, 0] = np.mean(y[-pred_len:])
    #     return pred_residual

    # def _predict_residual(self, pred_len):
    #     pred_residual = np.zeros((self.residual.shape[0], pred_len, 1))
    #     for i in range(self.residual.shape[0]):
    #         y = self.residual[i, :, 0]
    #         # 使用滑动平均模型进行预测
    #         model = ARMA(y, order=(0, 1))  # MA(1) 模型
    #         model_fit = model.fit(disp=False)
    #         forecast = model_fit.forecast(steps=pred_len)[0]
    #         pred_residual[i, :, 0] = forecast
    #     return pred_residual

    def _predict_residual(self, pred_len):
        pred_residual = np.zeros((self.residual.shape[0], pred_len, 1))
        for i in range(self.residual.shape[0]):
            y = self.residual[i, :, 0]
            # 使用最后一个窗口的均值进行预测
            mean_value = np.mean(y[-pred_len:])
            pred_residual[i, :, 0] = mean_value
        return pred_residual


seq_len = 96 * 6
pred_len = 96
# dataset = EttHour(root_path='../_datasets/ts-data/ETT-small/', data_path='ETTh1.csv')
# dataset = Electricity(root_path='../_datasets/ts-data/electricity/', data_path='electricity.csv')
dataset = Weather(root_path='../_datasets/ts-data/weather/', data_path='weather.csv')
mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
    'train', 'OT', seq_len, Augmentor('none', False), 3, 3
custom_dataset = CustomDataset(dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample)
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
history, label = next(iter(dataloader))  # batch, time, feature
# scaler = dataset.get_train_scaler('standard', target_column)
# history = scaler.transform(history.numpy().reshape(-1, 1)).reshape(batch_size, seq_len, 1)
# label = scaler.transform(label.numpy().reshape(-1, 1)).reshape(batch_size, pred_len, 1)
# seqs = history.copy()

# 对每个 batch 分别计算均值并进行缩放
history_transformed = np.zeros_like(history)
label_transformed = np.zeros_like(label)
scalers = []

for i in range(batch_size):
    scaler = StandardScaler()
    history_batch = history[i].numpy().reshape(-1, 1)
    label_batch = label[i].numpy().reshape(-1, 1)

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

periods = [60, 24, 30, 7, 12, 365]
# component_for_model = [['trend'], ['season'], ['residual'],
#                        ['trend', 'season'], ['season', 'residual'], ['trend', 'residual'],
#                        ['all']]
component_for_model = ['trend', 'season', 'residual',
                       'trend+season', 'season+residual', 'trend+residual',
                       'all']

# periods = [7, 30, 365]
# targets = [['trend'], ['season'], ['residual'], ['trend', 'season', 'residual']]

model = Uni2ts('./Uni2ts/ckpt/small', 'cpu')
# preds = Uni2ts('./Uni2ts/ckpt/small', 'cpu').forcast(seqs, pred_len)
# preds = Chronos('./Chronos/ckpt/tiny', 'cpu').forcast(seqs, pred_len)
# preds = Timer('./Timer/ckpt/Building_timegpt_d1024_l8_new_full.ckpt', 'cpu').forcast(seqs, pred_len)

fig, axs = plt.subplots(len(periods), len(component_for_model), figsize=(20, 20))
for i, period in enumerate(periods):
    for j, target in enumerate(component_for_model):
        decomposer = Decomposer(period=period, component_for_model=target)
        preprocessed_data = decomposer.pre_process(history)
        preds = model.forcast(preprocessed_data, pred_len)
        postprocessed_data = decomposer.post_process(preds)

        pred_total = np.concatenate([history, postprocessed_data], axis=1)

        # TODO:
        # 1. Plot the original data
        # 2. Plot the predicted data

        axs[i, j].plot(pred_total[0, :, 0], label='Predicted Data', color='orange')
        axs[i, j].plot(history[0, :, 0], label='Original Data', color='blue')

        axs[i, j].set_title(f'Period={period}, Target={target}', fontsize=8)
        axs[i, j].legend()

# plt.tight_layout()
plt.show()
