import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.fft import ifft
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sktime.forecasting.fbprophet import Prophet
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import get_dataset, CustomDataset
from AnyTransform.model import get_model

# from fbprophet import Prophet

from pmdarima.preprocessing import FourierFeaturizer
from pmdarima import auto_arima

import pandas as pd
from statsmodels.tsa.x13 import x13_arima_analysis


def time_start():
    return time.time()


def log_time_delta(t, event_name):
    d = time.time() - t
    print(f"{event_name} time: {d}")


class Decomposer:
    def __init__(self, period, component_for_model, parallel=False):
        self.period = period
        self.component_for_model = component_for_model.split('+')
        self.all_component_flag = component_for_model == 'all'
        self.trend = None
        self.season = None
        self.residual = None
        self.parallel = parallel

    def pre_process(self, data):
        if self.all_component_flag:
            return data

        def _decompose_series1(series):
            # Period=365, Target=trend
            # Preprocess time: 2.588042974472046
            # Period=365, Target=season
            # Preprocess time: 0.3718416690826416
            # Period=365, Target=residual
            # Preprocess time: 0.3341329097747803
            # Period=60, Target=trend
            # Preprocess time: 0.09929013252258301
            # Period=60, Target=season
            # Preprocess time: 0.07264876365661621
            # Period=60, Target=residual
            # Preprocess time: 0.17017793655395508

            #
            stl = STL(series, period=self.period, seasonal=13)
            result = stl.fit()
            return result.trend, result.seasonal, result.resid

        def _decompose_series5(series):
            # Period=365, Target=trend
            # Preprocess time: 1.6404809951782227
            # Period=365, Target=season
            # Preprocess time: 0.0523838996887207
            # Period=365, Target=residual
            # Preprocess time: 0.05175185203552246
            # Period=60, Target=trend
            # Preprocess time: 0.051614999771118164
            # Period=60, Target=season
            # Preprocess time: 0.05163908004760742
            # Period=60, Target=residual
            # Preprocess time: 0.0512700080871582
            freqs = np.fft.fftfreq(len(series))
            fft_values = fft(series)
            trend_fft = fft_values.copy()
            trend_fft[np.abs(freqs) > 1 / self.period] = 0
            trend = ifft(trend_fft).real

            # Compute seasonal component using FFT high frequency components
            season_fft = fft_values.copy()
            season_fft[np.abs(freqs) <= 1 / self.period] = 0
            season = ifft(season_fft).real

            # Compute residual
            residual = series - trend - season
            return trend, season, residual

        def _decompose_series6(series):
            # Compute trend using linear regression
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series, 1)
            trend = np.polyval(coeffs, x)

            # Compute seasonal component by subtracting the trend and detrending the result
            detrended = series - trend
            # season = self._compute_seasonal(detrended, self.period)
            season = np.zeros_like(data)
            period = self.period
            for i in range(period):
                seasonal_mean = np.mean(data[i::period])
                season[i::period] = seasonal_mean

            # Compute residual
            residual = series - trend - season
            return trend, season, residual

        if self.parallel:
            # 使用 joblib 并行处理每个批次的数据
            results = Parallel(n_jobs=30)(delayed(_decompose_series1)(data[i, :, 0]) for i in range(data.shape[0]))
            # 将结果分解成趋势、季节性和残差
            trends, seasons, residuals = zip(*results)
        else:
            trends, seasons, residuals = [], [], []
            for i in range(data.shape[0]):
                series = data[i, :, 0]  # Extract the series for each batch
                trend, seasonal, residual = _decompose_series1(series)
                # stl = STL(series, period=self.period, seasonal=13)
                # result = stl.fit()
                trends.append(trend)
                seasons.append(seasonal)
                residuals.append(residual)
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

    # def _predict_trend(self, pred_len):
    #     pred_trend = np.zeros((self.trend.shape[0], pred_len, 1))
    #     for i in range(self.trend.shape[0]):
    #         y = self.trend[i, :, 0]
    #         x = np.arange(len(y))
    #         future_x = np.arange(len(y), len(y) + pred_len)
    #         pred_trend[i, :, 0] = np.interp(future_x, x, y)
    #     return pred_trend

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

dataset = get_dataset('Weather')
mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
    'train', 'OT', seq_len, Augmentor('none', False), 200, 200
custom_dataset = CustomDataset(dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample)
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
idx, history, label = next(iter(dataloader))  # batch, time, feature
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

# periods = [60, 24, 30, 7, 12, 365]
# component_for_model = [['trend'], ['season'], ['residual'],
#                        ['trend', 'season'], ['season', 'residual'], ['trend', 'residual'],
#                        ['all']]

periods = [365, 60]
component_for_model = ['trend', 'season', 'residual']

# periods = [7, 30, 365]
# targets = [['trend'], ['season'], ['residual'], ['trend', 'season', 'residual']]

model = get_model('Timer-LOTSA', 'cpu')
# preds = Uni2ts('./Uni2ts/ckpt/small', 'cpu').forcast(seqs, pred_len)
# preds = Chronos('./Chronos/ckpt/tiny', 'cpu').forcast(seqs, pred_len)
# preds = Timer('./Timer/ckpt/Building_timegpt_d1024_l8_new_full.ckpt', 'cpu').forcast(seqs, pred_len)

fig, axs = plt.subplots(len(periods), len(component_for_model), figsize=(20, 20))
for i, period in enumerate(periods):
    for j, target in enumerate(component_for_model):
        print(f'Period={period}, Target={target}')
        decomposer = Decomposer(period=period, component_for_model=target, parallel=True)
        t = time_start()
        preprocessed_data = decomposer.pre_process(history)
        log_time_delta(t, 'Preprocess')
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
