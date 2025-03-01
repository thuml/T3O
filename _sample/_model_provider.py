import argparse
import os
import random
import sys
from math import ceil

import numpy as np
import torch
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from statsmodels.tsa.stl._stl import STL
from tqdm import tqdm

from AnyTransform import ExpTimer
import matplotlib
import matplotlib.pyplot as plt

import plotly as py
import plotly.graph_objs as go

from statsmodels.tsa.seasonal import seasonal_decompose

from AnyTransform.dataset import Electricity


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None


class Timer:
    def __init__(self, ckpt_path, device):
        parser = argparse.ArgumentParser(description='TimesNet')

        # basic config
        parser.add_argument('--task_name', type=str, default='large_finetune')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--model', type=str, default='Timer')
        parser.add_argument('--ckpt_path', type=str,
                            default='checkpoints/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt')
        # model define
        parser.add_argument('--patch_len', type=int, default=96)
        parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=8, help='num of encoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--factor', type=int, default=3, help='attn factor')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true')
        # GPU
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        # args = parser.parse_args()

        args = argparse.Namespace(
            task_name='large_finetune',
            seed=0,
            model='Timer',
            ckpt_path=ckpt_path,
            patch_len=96,
            d_model=1024,
            n_heads=16,
            e_layers=8,
            d_ff=2048,
            factor=3,
            dropout=0.1,
            activation='gelu',
            output_attention=False,
            use_gpu=True,
            gpu=0,
            use_multi_gpu=False,
            devices='0,1,2,3'
        )

        # _, args = parser.parse_known_args()  # 只解析默认的参数，而不是从命令行解析 ...

        fix_seed = args.seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        args.ckpt_path = ckpt_path
        assert 'cpu' == device or 'cuda' in device
        args.use_gpu = True if 'cuda' in device else False
        args.gpu = device.split(':')[-1] if 'cuda' in device else 0
        print(f'args.use_gpu={args.use_gpu}, args.gpu={args.gpu}')

        self.args = args
        self.exp = ExpTimer(args)

    def ar_infer(self, data, step):
        # (n,1) -> (1,n,1)
        patch_len = self.args.patch_len  # 96
        seq_len = len(data)
        data = data.reshape(1, -1, 1)
        pred_total = self.exp.raw_inference_with_no_scaler(data, step)
        _pred_total = pred_total.reshape(-1)
        pred = _pred_total[seq_len:]
        assert len(pred) == step * patch_len, f'len(pred)={len(pred)}'
        return pred


import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import argparse
import random
import numpy as np


class Uni2TS:
    def __init__(self, ckpt_path, device):
        # 模型是用96预测96
        self.patch_size = 96
        self.model_seq_l = self.patch_size
        self.model_pred_l = self.patch_size
        self.batch_size = 1
        self.model_size = 'small'
        self.device = device
        self.num_samples = 100  # 多次预测取median...
        self.ckpt_path = os.path.relpath(ckpt_path)
        print(os.path.exists(self.ckpt_path), self.ckpt_path)
        self.model = MoiraiForecast(
            # module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{self.model_size}"),
            module=MoiraiModule.from_pretrained(ckpt_path, local_files_only=True),
            prediction_length=self.model_pred_l,
            context_length=self.model_seq_l,
            patch_size='auto',
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self.predictor = self.model.create_predictor(batch_size=self.batch_size, device=self.device)

    def ar_infer(self, data, step):
        assert type(data) == np.ndarray and data.ndim == 1, \
            f'type(data)={type(data)}, data.ndim={data.ndim}'
        real_pred_l = step * self.model_pred_l
        seq_with_zero_pred = np.concatenate([data, np.zeros(real_pred_l)])  # !!!!!!!!!!! 他的predict钱的split需要gt！
        data_pd = pd.DataFrame(seq_with_zero_pred)
        print('data_pd.head()', data_pd.head())
        start_date = '2000-01-01'
        date_range = pd.date_range(start=start_date, periods=len(data_pd), freq='D')
        data_pd.index = date_range
        print('data_pd.head()', data_pd.head())

        ds = PandasDataset(dict(data_pd))
        train, test_template = split(ds, offset=-real_pred_l)
        test_data = test_template.generate_instances(
            prediction_length=self.patch_size,
            windows=real_pred_l // self.patch_size,  # ..
            distance=self.patch_size,  # ..
        )
        forecasts = self.predictor.predict(test_data.input)
        pred = None
        # pred由forecast的sample拼接而成
        for forecast in forecasts:
            pred = forecast.quantile(0.5) if pred is None else np.concatenate([pred, forecast.quantile(0.5)])
        assert len(pred) == real_pred_l, f'len(pred)={len(pred)}'
        return pred




        # forecast_list = list(forecasts)
        # assert len(forecast_list) == 1, f'len(forcast_list)={len(forecast_list)}'
        # forecast = forecast_list[0]
        # print('forecast', forecast)
        # print('forecast.samples', forecast.samples)
        # print('forecast.samples.shape', forecast.samples.shape)
        # print('forecast.quantile(0.5)', forecast.quantile(0.5))
        #
        # pred = forecast.quantile(0.5)  # median
        # assert len(pred) == self.model_pred_l, f'len(pred)={len(pred)}'
        # return pred


# Example usage
if __name__ == "__main__":
    ckpt_path = './Uni2ts/ckpt/small'
    uni2ts = Uni2TS(ckpt_path, 'cpu')
    # url = (
    #     "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
    #     "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
    # )
    # df = pd.read_csv('ts_wide.csv', index_col=0, parse_dates=True)
    # df.to_csv('ts_wide.csv')
    # print(df)
    # exit()

    patch_l = 96
    seq_len = 96 * 3
    pred_len = 96 * 2
    dataset = Electricity(root_path='../_datasets/ts-data/electricity/', data_path='electricity.csv')
    seq = dataset.np_data_dict['OT'][dataset.train_start:dataset.train_end][:seq_len]
    truth_total = dataset.np_data_dict['OT'][dataset.train_start:dataset.train_end][:seq_len + pred_len]

    # df = pd.read_csv('AnyTransform/ts_wide.csv', index_col=0, parse_dates=True)
    # seq = np.array(df['A'][:seq_len].values)
    # truth_total = np.array(df['A'][:seq_len + pred_len].values)

    print('len(truth_total)', len(truth_total))
    print('len(seq)', len(seq))

    pred = uni2ts.ar_infer(seq, pred_len // patch_l)
    print('len(pred)', len(pred))
    # pred = Timer('checkpoints/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt', 'cpu').ar_infer(seq, 1)

    # 画个图吧
    pred_total = np.concatenate([seq, pred])
    plt.plot(pred_total, label='pred', color='orange')
    plt.plot(truth_total, label='truth', color='blue')
    plt.legend()
    plt.show()
