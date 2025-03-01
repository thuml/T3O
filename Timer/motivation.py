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

from exp.exp_large_few_shot_roll_demo import Exp_Large_Few_Shot_Roll_Demo
from utils.metrics import metric
import matplotlib
import matplotlib.pyplot as plt

import plotly as py
import plotly.graph_objs as go

from statsmodels.tsa.seasonal import seasonal_decompose

np.set_printoptions(precision=3)


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, default='large_finetune')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model', type=str, default='Timer')
parser.add_argument('--ckpt_path', type=str, default='checkpoints/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt')

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

args = parser.parse_args()
fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.ckpt_path = 'ckpt/Building_timegpt_d1024_l8_new_full.ckpt'
args.use_gpu = True if torch.cuda.is_available() else False

exp = Exp_Large_Few_Shot_Roll_Demo(args)


def get_ETTh1_HUFL_data():
    # out:[1,n,1]
    file_path = "../_datasets/ts-data/ETT-small/ETTh1.csv"
    # data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    # ValueError: could not convert string '2016-07-01 00:00:00' to float64 at row 0, column 1.
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1, usecols=(1, 2, 3, 4, 5, 6, 7))
    print(data[0])
    data = data[:, 1]
    data = np.expand_dims(data, 0)
    data = np.expand_dims(data, 2)
    return data


def get_ETTm1_HULL_data():
    # out:[1,n,1]
    file_path = "../_datasets/ts-data/ETT-small/ETTm1.csv"
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1, usecols=(1, 2, 3, 4, 5, 6, 7))
    data = data[:, 2]
    data = np.expand_dims(data, 0)
    data = np.expand_dims(data, 2)
    return data


def get_ETTm2_data(column='HUFL'):
    # out:[1,n,1]
    # date  HUFL	HULL	MUFL	MULL	LUFL	LULL	OT
    column_list = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    file_path = "../_datasets/ts-data/ETT-small/ETTm1.csv"
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1, usecols=(1, 2, 3, 4, 5, 6, 7))
    data = data[:, column_list.index(column)]
    print(data[0:10])
    data = np.expand_dims(data, 0)
    data = np.expand_dims(data, 2)
    return data


def moti1(column_name):
    print("###############################################################################################")
    print("moti1:", column_name)
    if not os.path.exists("./motivation/"):
        os.makedirs("./motivation")
    res_dir = os.path.join('./motivation', column_name)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    patch_len = args.patch_len  # 96
    # date  HUFL	HULL	MUFL	MULL	LUFL	LULL	OT
    data = get_ETTm2_data(column=column_name)
    print("data.shape", data.shape)  # 15678 # 60000
    split_ratio_list = np.linspace(13 / 16, 15 / 16, 60)
    # np.linspace(13 / 16, 15 / 16, 30)  # 重复测试，多了反而不懂幅度巨大。。。。[13 / 16, 14 / 16, 15 / 16] [15 / 16]
    split_idx_list = [int(data.shape[1] * r) for r in split_ratio_list]
    print("split_idx_list:", split_idx_list)
    # seq: data[:, split_idx-seq_len:split_idx,:], pred: data[:, split_idx:split_idx+pred_len,:]

    # pred_len_list = [96, 192, 288, 384, 480]
    pred_patch_ratios = [1, 3, 5]
    # seq_patch_ratio: 1,100 一共20个点,先测5个点吧
    seq_patch_ratios = list(range(1, 100, 1))
    print("pred_patch_ratios:", pred_patch_ratios)
    print("seq_patch_ratios:", seq_patch_ratios)

    if os.path.exists(os.path.join(res_dir, f'result_dict_dict.npy')):
        result_dict_dict = np.load(os.path.join(res_dir, f'result_dict_dict.npy'), allow_pickle=True).item()
    else:
        result_dict_dict = {}  # pred_{pred_patch_ratio}_seq_{seq_patch_ratio} -> result_dict
        # -> {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "mspe": mspe}
        for pred_patch_r in pred_patch_ratios:
            pred_len = pred_patch_r * patch_len
            min_mse = np.inf
            # for i in tqdm(range(100), desc='Processing'):
            # for seq_patch in seq_patch_ratio:
            bar = tqdm(seq_patch_ratios, desc='Processing', ncols=100)  # 一次一分钟
            for seq_patch_r in bar:
                seq_len = seq_patch_r * patch_len
                result_dict_dict[f"pred_{pred_patch_r}_seq_{seq_patch_r}"] = {}
                for split_idx in split_idx_list:
                    seq = data[:, split_idx - seq_len:split_idx, :]
                    pred_total = exp.raw_inference(seq.copy(), pred_patch_r)  # [1, seq_len+pred_len, 1]
                    assert pred_total.shape[1] == seq_len + pred_len, \
                        f"pred_total.shape[1]={pred_total.shape[1]}, seq_len+pred_len={seq_len + pred_len}"
                    pred_line = pred_total[0, :, 0]
                    truth_line = data[0, split_idx - seq_len:split_idx + pred_len, 0]
                    assert len(pred_line) == len(truth_line), \
                        f"len(pred_line)={len(pred_line)}, len(truth_line)={len(truth_line)}"
                    mae, mse, rmse, mape, mspe = metric(pred_line[-pred_len:], truth_line[-pred_len:])
                    # print(f"seq_len={seq_len}, pred_len={pred_len}, "
                    #       f"mae={mae}, mse={mse}, rmse={rmse}, mape={mape}, mspe={mspe}")
                    bar.set_postfix(mse=mse, rmse=rmse, mape=mape, mspe=mspe)
                    result_dict_dict[f"pred_{pred_patch_r}_seq_{seq_patch_r}"][f"split_idx_{split_idx}"] = \
                        {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}
                    if mse < min_mse:
                        min_mse = mse
                        print(f"pred_patch_ratio={pred_patch_r}, seq_patch_ratio={seq_patch_r}, "
                              f"split_idx={split_idx}, mae={mae}, mse={mse}, rmse={rmse}, mape={mape}, mspe={mspe}")
                        # 保存pred_patch最好的结果
                        np.save(os.path.join(res_dir, f'pred_{pred_patch_r}_seq_{seq_patch_r}_pred.npy'), pred_line)
                        np.save(os.path.join(res_dir, f'pred_{pred_patch_r}_seq_{seq_patch_r}_truth.npy'), truth_line)
                        # 画个图吧
                        plt.figure(figsize=(len(truth_line) // patch_len * 5, 5))
                        plt.plot(pred_line, label="pred")
                        plt.plot(truth_line, label="truth")
                        plt.legend()
                        plt.savefig(os.path.join(res_dir, f'pred_{pred_patch_r}_seq_{seq_patch_r}.pdf'),
                                    bbox_inches='tight')
        # 保存result_dict_dict到本地
        np.save(os.path.join(res_dir, f'result_dict_dict.npy'), result_dict_dict)

    # 整体画图：
    # 3个直方图 pred_patch_ratio = 1,3,5
    # x: seq_patch_ratio, 1-100
    # y: num_point
    # 每个柱子的高度h：在确定的pred_patch_ratio的场景下，能得到最小的mse的seq_patch_ratio的split_idx样本数量为h
    top_ratio = 25 / 100
    for pred_patch_r in pred_patch_ratios:
        plt.figure(figsize=(12, 5))
        x = seq_patch_ratios
        y = [0] * len(x)
        for split_idx in split_idx_list:
            mse_list = [0] * len(seq_patch_ratios)
            for seq_patch_r in seq_patch_ratios:
                v = result_dict_dict[f"pred_{pred_patch_r}_seq_{seq_patch_r}"][f"split_idx_{split_idx}"]["mse"]
                mse_list[seq_patch_ratios.index(seq_patch_r)] = v
            # mse的百分比排名在25%的可以+1
            mse_list = np.array(mse_list)
            mse_list = np.argsort(mse_list)  # 小的会排在前面
            mse_list = mse_list[:int(len(mse_list) * top_ratio)]
            for seq_patch_r in mse_list:
                y[seq_patch_r] += 1

        plt.bar(x, y, width=1)
        plt.title(f"pred_patch_ratio={pred_patch_r}, top_ratio={top_ratio}")
        plt.xlabel("seq_patch_ratio")
        plt.ylabel("num_point")
        plt.savefig(os.path.join(res_dir, f'pred_patch_ratio_{pred_patch_r}_top_ratio_{top_ratio}.pdf'),
                    bbox_inches='tight')
        plt.close()


def moti2():
    # 探索不同的scale方法对结果的影响
    patch_len = args.patch_len  # 96
    # data = get_ETTm1_HULL_data()
    data = get_ETTm2_data(column='LUFL')
    print("data.shape", data.shape)

    split_ratio_list = np.linspace(1 / 2, 7 / 8, 60)
    split_idx_list = [int(data.shape[1] * r) for r in split_ratio_list]

    # pred_patch_ratios = [1, 3, 5]
    # seq_patch_ratios = list(range(1, 100, 1))
    pred_patch_r = 6
    seq_patch_r = 3

    pred_len = patch_len * pred_patch_r
    seq_len = patch_len * seq_patch_r

    scale_methods = ['standard', 'maxabs', 'robust', None]
    # （minmax/std/rob/log-10/...）
    # 'minmax'效果很差，影响结果展示了
    # scale_methods = ['standard', None]

    # scale_methods = [None, 'minmax_0_1', 'minmax_-1_1', 'minmax_-0.5_0.5',
    #                  'robust_25_75', 'robust_10_90', 'robust_1_99']

    res_dir = os.path.join('./motivation', 'scale')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    result_dict_dict = {}  #

    for scale_method in scale_methods:
        for split_idx in split_idx_list:
            seq = data[:, split_idx - seq_len:split_idx, :]
            # scale -> model -> scale_back!
            if scale_method is None:
                scaler = None
            elif scale_method == 'standard':
                scaler = StandardScaler()
                seq = scaler.fit_transform(seq[0])  # (n_samples, n_features)=(seq_len, 1) ok
                seq = seq[np.newaxis, :]  # (seq_len, 1) -> (1, seq_len, 1)
            elif scale_method == 'minmax':
                scaler = MinMaxScaler()
                seq = scaler.fit_transform(seq[0])
                seq = seq[np.newaxis, :]
            elif scale_method == 'maxabs':
                scaler = MaxAbsScaler()
                seq = scaler.fit_transform(seq[0])
                seq = seq[np.newaxis, :]
            elif scale_method == 'robust':
                scaler = RobustScaler()
                seq = scaler.fit_transform(seq[0])
                seq = seq[np.newaxis, :]
            else:
                raise ValueError(f"scale_method={scale_method}")
            assert seq.shape[1] == seq_len and seq.shape[2] == 1 and seq.shape[0] == 1, f"seq.shape={seq.shape}"
            # _pred_total = exp.raw_inference(seq.copy(), pred_patch_r)
            # assert _pred_total.shape[1] == seq_len + pred_len, \
            #     f"_pred_total.shape[1]={_pred_total.shape[1]}, seq_len+pred_len={seq_len + pred_len}"
            # pred_total = scaler.inverse_transform(_pred_total[0]).reshape(1, -1, 1) \
            #     if scaler is not None else _pred_total
            if scale_method is None:
                pred_total = exp.raw_inference(seq.copy(), pred_patch_r)
            else:
                _pred_total = exp.raw_inference_with_no_scaler(seq.copy(), pred_patch_r)
                pred_total = scaler.inverse_transform(_pred_total[0]).reshape(1, -1, 1)
            pred_line = pred_total[0, :, 0]
            # scale back
            truth_line = data[0, split_idx - seq_len:split_idx + pred_len, 0]
            assert len(pred_line) == len(truth_line), \
                f"len(pred_line)={len(pred_line)}, len(truth_line)={len(truth_line)}"
            mae, mse, rmse, mape, mspe = metric(pred_line[-pred_len:], truth_line[-pred_len:])
            result_dict_dict[f"scale_method_{scale_method}_split_idx_{split_idx}"] = \
                {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}
            print(f"scale_method={scale_method}, split_idx={split_idx}, "
                  f"mae={mae}, mse={mse}, rmse={rmse}, mape={mape}, mspe={mspe}")
        # 画个图吧
        plt.figure(figsize=(len(truth_line) // patch_len * 5, 5))
        plt.plot(pred_line, label="pred")
        plt.plot(truth_line, label="truth")
        plt.legend()
        plt.savefig(os.path.join(res_dir, f'scale_method_{scale_method}_split_idx_{split_idx}.pdf'),
                    bbox_inches='tight')
    np.save(os.path.join(res_dir, f'result_dict_dict.npy'), result_dict_dict)
    # 画个图吧：Bar：x->method, y->mean_mse
    mean_mse_list = []
    for scale_method in scale_methods:
        mse_list = [result_dict_dict[f"scale_method_{scale_method}_split_idx_{split_idx}"]["mse"]
                    for split_idx in split_idx_list]
        mean_mse = np.mean(mse_list)
        mean_mse_list.append(mean_mse)
    plt.figure(figsize=(8, 6))
    scale_methods_names = [x if x is not None else 'None' for x in scale_methods]
    plt.bar(scale_methods_names, mean_mse_list)
    plt.xlabel("scale_method")
    plt.ylabel("mean_mse")
    plt.savefig(os.path.join(res_dir, f'mean_mse.pdf'), bbox_inches='tight')


def moti3(column, seq_patch_r, pred_patch_r):
    # 考虑不同的sample对结果的影响 （上采样原始数据，让模型输入数据更小）
    # data = get_ETTm2_data(column='HUFL')
    data = get_ETTm2_data(column=column)
    print("data.shape", data.shape)

    patch_len = args.patch_len  # 96
    # seq_patch_r = 30
    # pred_patch_r = 10

    pred_len = patch_len * pred_patch_r
    seq_len = patch_len * seq_patch_r

    split_ratio_list = np.linspace(1 / 2, 7 / 8, 60)  # 60
    split_idx_list = [int(data.shape[1] * r) for r in split_ratio_list]

    from scipy import signal
    # 不同的下采样倍数
    # sample_ratio = [1, 2, 3, 4, 5]  # 长度为 1/sample_ratio
    sample_ratio = np.linspace(1, 10, 30)

    res_dir = os.path.join('./motivation', '_sample', f"{column}", f"seq_{seq_patch_r}_pred_{pred_patch_r}")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    result_dict_dict = {}  #

    for sample_r in sample_ratio:
        for split_idx in split_idx_list:
            seq = data[:, split_idx - seq_len:split_idx, :]
            # down _sample data -> model input -> model output -> up _sample output
            model_seq_len = ceil(seq_patch_r / sample_r) * patch_len
            model_pred_patch_r = ceil(pred_patch_r / sample_r)
            model_pred_len = patch_len * model_pred_patch_r
            # ##################### begin
            _seq = seq[0, :, 0]
            _seq = signal.resample(_seq, ceil(seq_len / sample_r))  # ceil(seq_len / sample_r)重要!!!!!! 不是整数倍
            # 长度不够patch整数倍的话要在前面补0
            _seq = np.concatenate((np.zeros(patch_len - len(_seq) % patch_len), _seq)) \
                if len(_seq) % patch_len != 0 else _seq
            _seq = _seq[np.newaxis, :, np.newaxis]
            assert _seq.shape[1] % patch_len == 0 and _seq.shape[2] == 1 and _seq.shape[0] == 1, \
                f"_seq.shape={_seq.shape}"
            # print(_seq[0, :5, 0])
            _pred_total = exp.raw_inference(_seq.copy(), model_pred_patch_r)
            assert _pred_total.shape[1] == model_seq_len + model_pred_len, \
                f"_pred_total.shape[1]={_pred_total.shape[1]}, " \
                f"model_seq_len+model_pred_len={model_seq_len + model_pred_len}"
            _pred = _pred_total[:, -model_pred_len:, :]
            # print(_pred[0, :5, 0])
            _pred = signal.resample(_pred[0, :, 0], ceil(model_pred_len * sample_r))  # #### 当心有错！！！！scale相似的问题？
            assert pred_len <= len(_pred), f"pred_len={pred_len}, len(_pred)={len(_pred)}"
            # ##################### end
            pred = _pred[:pred_len].reshape(1, -1, 1)
            line_pred = np.concatenate((seq[0, :, 0], pred[0, :, 0]), axis=0)
            line_truth = data[0, split_idx - seq_len:split_idx + pred_len, 0]
            assert len(line_pred) == len(line_truth), \
                f"len(line_pred)={len(line_pred)}, len(line_truth)={len(line_truth)}"
            mae, mse, rmse, mape, mspe = metric(line_pred[-pred_len:], line_truth[-pred_len:])
            result_dict_dict[f"sample_r_{sample_r}_split_idx_{split_idx}"] = \
                {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}
            print(f"sample_r={sample_r}, split_idx={split_idx}, "
                  f"mae={mae}, mse={mse}, rmse={rmse}, mape={mape}, mspe={mspe}")
        # 画个图吧
        plt.figure(figsize=(len(line_truth) // patch_len * 5, 5))
        plt.plot(line_pred, label="pred")
        plt.plot(line_truth, label="truth")
        plt.legend()
        plt.savefig(os.path.join(res_dir, f'sample_r_{sample_r}_split_idx_{split_idx}.pdf'),
                    bbox_inches='tight')
    np.save(os.path.join(res_dir, f'result_dict_dict.npy'), result_dict_dict)
    # 画个图吧
    # Bar: x=sample_r y=mean_mse
    mean_mse_list = []
    for sample_r in sample_ratio:
        mse_list = [result_dict_dict[f"sample_r_{sample_r}_split_idx_{split_idx}"]["mse"]
                    for split_idx in split_idx_list]
        mean_mse = np.mean(mse_list)
        mean_mse_list.append(mean_mse)
    plt.figure(figsize=(8, 6))
    plt.bar(sample_ratio, mean_mse_list, width=0.1)
    plt.xlabel("sample_r")
    plt.ylabel("mean_mse")
    plt.savefig(os.path.join(res_dir, f'mean_mse.pdf'), bbox_inches='tight')
    # 画个图吧
    # histigram：x=sample_r y=num_point
    # 对于不同的idx，找到mse小的排名靠前的sample_r，y[sample_r]+1
    rate_ratio = 0.25
    y = [0] * len(sample_ratio)
    for split_idx in split_idx_list:
        mse_list = []
        for sample_r in sample_ratio:
            mse_list.append(result_dict_dict[f"sample_r_{sample_r}_split_idx_{split_idx}"]["mse"])
        mse_list = np.array(mse_list)
        mse_list = np.argsort(mse_list)
        mse_list = mse_list[:int(len(mse_list) * rate_ratio)]
        for sample_r in mse_list:
            y[sample_r] += 1
    plt.figure(figsize=(8, 6))
    plt.bar(sample_ratio, y, width=0.1)
    plt.xlabel("sample_r")
    plt.ylabel("num_point")
    plt.savefig(os.path.join(res_dir, f'num_point.pdf'), bbox_inches='tight')
    #
    # from scipy import interpolate
    # interpolate.interp1d(kind=)


def moti4():
    # 尝试用FFT降噪

    patch_len = args.patch_len  # 96
    # data = get_ETTm2_data(column='HUFL')
    data = get_ETTm2_data(column='LULL')
    print("data.shape", data.shape)

    split_ratio_list = np.linspace(1 / 2, 7 / 8, 60)
    split_idx_list = [int(data.shape[1] * r) for r in split_ratio_list]

    seq_patch_r = 3
    pred_patch_r = 1

    pred_len = patch_len * pred_patch_r
    seq_len = patch_len * seq_patch_r

    # # 进行傅里叶变换
    # fft_result = np.fft.fft(noisy_signal)  # 傅里叶变换
    # # 获取频率轴
    # freqs = np.fft.fftfreq(t.size, t[1] - t[0])
    # print(freqs)  # 等间距的 最大100 = 1 / interval / 2 = F / 2 = Nyquist Frequency ！！！！！
    # print(max(freqs), len(freqs), len(noisy_signal), t.size, t[1] - t[0])
    # # 频谱滤波：去除高频噪声
    # cutoff_freq = max(abs(freqs)) / 1000  # 截止频率 cutoff_freq越小，去掉的高频越多，越平滑
    # fft_result[np.abs(freqs) > cutoff_freq] = 0  # 将高于截止频率的频率成分置零
    # # 反向傅里叶变换得到滤波后的信号
    # filtered_signal = np.fft.ifft(fft_result).real

    cutoff_ratios = np.linspace(0, 0.9, 30)
    # cutoff_freq = max(abs(freqs)) * (1 - cutoff_ratio)
    # # cutoff_ratio越大，cutoff_freq越小，去掉的cutoff_freq越小以上的高频越多，越平滑

    res_dir = os.path.join('./motivation', 'fft')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    result_dict_dict = {}  #

    for cutoff_ratio in cutoff_ratios:
        for split_idx in split_idx_list:
            seq = data[:, split_idx - seq_len:split_idx, :]
            #  ##################### begin
            _seq = seq[0, :, 0]
            _seq_f = np.fft.fft(_seq)
            freqs = np.fft.fftfreq(_seq_f.size, 1)
            cutoff_freq = max(abs(freqs)) * (1 - cutoff_ratio)  # cutoff_ratio越大，去掉的高频越多，越平滑
            _seq_f[np.abs(freqs) > cutoff_freq] = 0
            _seq = np.fft.ifft(_seq_f).real
            _seq = _seq[np.newaxis, :, np.newaxis]
            assert _seq.shape[1] == seq_len and _seq.shape[2] == 1 and _seq.shape[0] == 1, f"_seq.shape={_seq.shape}"
            # print(_seq[0, :5, 0])
            _pred_total = exp.raw_inference(_seq.copy(), pred_patch_r)
            assert _pred_total.shape[1] == seq_len + pred_len, \
                f"_pred_total.shape[1]={_pred_total.shape[1]}, seq_len+pred_len={seq_len + pred_len}"
            _pred = _pred_total[:, -pred_len:, :]
            # print(_pred[0, :5, 0])
            pred = np.concatenate((_seq[0, :, 0], _pred[0, :, 0]), axis=0)
            truth = data[0, split_idx - seq_len:split_idx + pred_len, 0]
            assert len(pred) == len(truth), f"len(pred)={len(pred)}, len(truth)={len(truth)}"
            mae, mse, rmse, mape, mspe = metric(pred[-pred_len:], truth[-pred_len:])
            result_dict_dict[f"cutoff_ratio_{cutoff_ratio}_split_idx_{split_idx}"] = \
                {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}
            print(f"cutoff_ratio={cutoff_ratio}, split_idx={split_idx}, "
                  f"mae={mae}, mse={mse}, rmse={rmse}, mape={mape}, mspe={mspe}")
        # 画个图吧
        plt.figure(figsize=(len(truth) // patch_len * 5, 5))
        plt.plot(pred, label="pred")
        plt.plot(truth, label="truth")
        plt.legend()
        plt.savefig(os.path.join(res_dir, f'cutoff_ratio_{cutoff_ratio}_split_idx_{split_idx}.pdf'),
                    bbox_inches='tight')
    np.save(os.path.join(res_dir, f'result_dict_dict.npy'), result_dict_dict)
    # 画个图吧
    # Bar: x=cutoff_ratio y=mean_mse
    mean_mse_list = []
    for cutoff_ratio in cutoff_ratios:
        mse_list = [result_dict_dict[f"cutoff_ratio_{cutoff_ratio}_split_idx_{split_idx}"]["mse"]
                    for split_idx in split_idx_list]
        mean_mse = np.mean(mse_list)
        mean_mse_list.append(mean_mse)
    plt.figure(figsize=(8, 6))
    plt.bar(cutoff_ratios, mean_mse_list, width=0.01)
    plt.xlabel("cutoff_ratio")
    plt.ylabel("mean_mse")
    plt.savefig(os.path.join(res_dir, f'mean_mse.pdf'), bbox_inches='tight')
    # 画个图吧
    # histigram：x=cutoff_ratio y=num_point
    # 对于不同的idx，找到mse小的排名靠前的cutoff_ratio，y[cutoff_ratio]+1
    rate_ratio = 0.25
    y = [0] * len(cutoff_ratios)
    for split_idx in split_idx_list:
        mse_list = []
        for cutoff_ratio in cutoff_ratios:
            mse_list.append(result_dict_dict[f"cutoff_ratio_{cutoff_ratio}_split_idx_{split_idx}"]["mse"])
        mse_list = np.array(mse_list)
        mse_list = np.argsort(mse_list)
        mse_list = mse_list[:int(len(mse_list) * rate_ratio)]
        for cutoff_ratio in mse_list:
            y[cutoff_ratio] += 1
    plt.figure(figsize=(8, 6))
    plt.bar(cutoff_ratios, y, width=0.01)
    plt.xlabel("cutoff_ratio")
    plt.ylabel("num_point")
    plt.savefig(os.path.join(res_dir, f'num_point.pdf'), bbox_inches='tight')


def moti5():
    # 使用STL分解之后，对每个分量用Timer进行预测，然后合并，跟Truth比较，跟不分解直接预测的Timer比较
    # 画五个图
    # 图1：truth<->infer(all)：seq_len + pred_len
    # 图2：truth<->infer(trend)：seq_len + pred_len
    # 图3：truth<->infer(seasonal)：seq_len + pred_len
    # 图4：truth<->infer(residual)：seq_len + pred_len
    # 图5：truth<->infer(trend)+infer(seasonal)+infer(residual)：seq_len + pred_len

    # data = get_ETTm2_data(column='LULL')
    # data = get_ETTm2_data(column='HULL')
    data = get_ETTm2_data(column='HULL')
    print("data.shape", data.shape)

    patch_len = args.patch_len  # 96
    seq_patch_r = 20
    pred_patch_r = 1

    pred_len = patch_len * pred_patch_r
    seq_len = patch_len * seq_patch_r

    split_ratio_list = np.linspace(1 / 2, 7 / 8, 10)  # ###
    split_idx_list = [int(data.shape[1] * r) for r in split_ratio_list]

    res_dir = os.path.join('./motivation', 'stl')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    result_dict_dict = {}  #

    dec_methods = [None, 'trend', 'trend+seasonal', 'trend+seasonal+residual']
    # ValueError: period must be a positive integer >= 2
    dec_periods = [12, 24, 30, 48, 60, 96]
    # 15min * 4 * 24 = 24h!!!!! # 跟数据集有关。。。
    # 现象：period越小，trend分解出的周期性越明显
    # dec_periods = np.arange(2, 12, 2)  # 1-9

    for dec_method in dec_methods:
        for dec_period in dec_periods:
            for split_idx in split_idx_list:
                seq = data[:, split_idx - seq_len:split_idx, :]
                # decompose
                _seq = seq[0, :, 0]
                decomp = STL(_seq, period=dec_period).fit()
                # decomp = seasonal_decompose(seq, period=dec_period)
                assert len(decomp.trend) == len(decomp.seasonal) == len(decomp.resid) == len(_seq), \
                    f"len(decomp.trend)={len(decomp.trend)}, len(decomp.seasonal)={len(decomp.seasonal)}, " \
                    f"len(decomp.resid)={len(decomp.resid)}, len(seq)={len(_seq)}"
                default_total = exp.raw_inference(_seq.reshape(1, -1, 1).copy(), pred_patch_r)
                trend_total = exp.raw_inference(decomp.trend.reshape(1, -1, 1).copy(), pred_patch_r)
                seasonal_total = exp.raw_inference(decomp.seasonal.reshape(1, -1, 1).copy(), pred_patch_r)
                residual_total = exp.raw_inference(decomp.resid.reshape(1, -1, 1).copy(), pred_patch_r)
                assert default_total.shape[1] == trend_total.shape[1] == seasonal_total.shape[1] == \
                       residual_total.shape[1] == seq_len + pred_len, \
                    f"default_total.shape[1]={default_total.shape[1]}, trend_total.shape[1]={trend_total.shape[1]}, " \
                    f"seasonal_total.shape[1]={seasonal_total.shape[1]}, residual_total.shape[1]={residual_total.shape[1]}"
                default_pred = default_total[0, -pred_len:, 0]
                trend_pred = trend_total[0, -pred_len:, 0]
                seasonal_pred = seasonal_total[0, -pred_len:, 0]
                residual_pred = residual_total[0, -pred_len:, 0]
                # cat seq and xxx_pred
                if dec_method is None:
                    pred_line = np.concatenate((_seq, default_pred), axis=0)
                elif dec_method == 'trend':
                    pred_line = np.concatenate((_seq, trend_pred), axis=0)
                elif dec_method == 'trend+seasonal':
                    pred_line = np.concatenate((_seq, trend_pred + seasonal_pred), axis=0)
                elif dec_method == 'trend+seasonal+residual':
                    pred_line = np.concatenate((_seq, trend_pred + seasonal_pred + residual_pred), axis=0)
                else:
                    raise ValueError(f"dec_method={dec_method}")
                truth_line = data[0, split_idx - seq_len:split_idx + pred_len, 0]
                assert len(pred_line) == len(truth_line), \
                    f"len(pred_line)={len(pred_line)}, len(truth_line)={len(truth_line)}"
                mae, mse, rmse, mape, mspe = metric(pred_line[-pred_len:], truth_line[-pred_len:])
                result_dict_dict[f"dec_method_{dec_method}_dec_period_{dec_period}_split_idx_{split_idx}"] = \
                    {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}
                print(f"dec_method={dec_method}, dec_period={dec_period}, split_idx={split_idx}, "
                      f"mae={mae}, mse={mse}, rmse={rmse}, mape={mape}, mspe={mspe}")
            # 画个图吧
            plt.figure(figsize=(len(truth_line) // patch_len * 5, 5))
            plt.plot(pred_line, label="pred")
            plt.plot(truth_line, label="truth")
            plt.legend()
            plt.savefig(
                os.path.join(res_dir, f'dec_method_{dec_method}_dec_period_{dec_period}_split_idx_{split_idx}.pdf'),
                bbox_inches='tight')
            # 画三个图吧，希望展示 trend/season/residual各自的预测效果
            plt.figure(figsize=(len(truth_line) // patch_len * 5, 5))
            plt.plot(trend_total[0, :, 0], label="trend_total")
            plt.plot(decomp.trend, label="decomp.trend")
            plt.legend()
            plt.savefig(
                os.path.join(res_dir,
                             f'dec_method_{dec_method}_dec_period_{dec_period}_split_idx_{split_idx}_trend.pdf'),
                bbox_inches='tight')
            plt.figure(figsize=(len(truth_line) // patch_len * 5, 5))
            plt.plot(seasonal_total[0, :, 0], label="seasonal_total")
            plt.plot(decomp.seasonal, label="decomp.seasonal")
            plt.legend()
            plt.savefig(
                os.path.join(res_dir,
                             f'dec_method_{dec_method}_dec_period_{dec_period}_split_idx_{split_idx}_seasonal.pdf'),
                bbox_inches='tight')
            plt.figure(figsize=(len(truth_line) // patch_len * 5, 5))
            plt.plot(residual_total[0, :, 0], label="residual_total")
            plt.plot(decomp.resid, label="decomp.residual")
            plt.legend()
            plt.savefig(
                os.path.join(res_dir,
                             f'dec_method_{dec_method}_dec_period_{dec_period}_split_idx_{split_idx}_residual.pdf'),
                bbox_inches='tight')

    np.save(os.path.join(res_dir, f'result_dict_dict.npy'), result_dict_dict)

    for dec_period in dec_periods:
        # 画个图吧
        # Bar: x=dec_method y=mean_mse
        mean_mse_list = []
        for dec_method in dec_methods:
            mse_list = [
                result_dict_dict[f"dec_method_{dec_method}_dec_period_{dec_period}_split_idx_{split_idx}"]["mse"]
                for split_idx in split_idx_list]
            mean_mse = np.mean(mse_list)
            mean_mse_list.append(mean_mse)
        plt.figure(figsize=(8, 6))
        dec_method_names = [x if x is not None else 'None' for x in dec_methods]
        plt.bar(dec_method_names, mean_mse_list, width=0.1)
        plt.title(f"dec_period={dec_period}")
        plt.xlabel("dec_method")
        plt.ylabel("mean_mse")
        plt.savefig(os.path.join(res_dir, f'dec_period_{dec_period}_mean_mse.pdf'), bbox_inches='tight')
        # 画个图吧
        # histigram：x=dec_method y=num_point
        # 对于不同的idx，找到mse小的排名靠前的dec_method，y[dec_method]+1
        rate_ratio = 0.25
        y = [0] * len(dec_methods)
        for split_idx in split_idx_list:
            mse_list = []
            for dec_method in dec_methods:
                mse_list.append(
                    result_dict_dict[f"dec_method_{dec_method}_dec_period_{dec_period}_split_idx_{split_idx}"]["mse"])
            mse_list = np.array(mse_list)
            mse_list = np.argsort(mse_list)
            mse_list = mse_list[:int(len(mse_list) * rate_ratio)]
            for dec_method in mse_list:
                y[dec_method] += 1
        plt.figure(figsize=(8, 6))
        dec_methods_names = [x if x is not None else 'None' for x in dec_methods]
        plt.bar(dec_methods_names, y, width=0.1)
        plt.title(f"dec_period={dec_period}")
        plt.xlabel("dec_method")
        plt.ylabel("num_point")
        plt.savefig(os.path.join(res_dir, f'dec_period_{dec_period}_num_point.pdf'), bbox_inches='tight')


def moti6():
    # 比较flip前后的预测情况差别

    data = get_ETTm2_data(column='HULL')
    # data = get_ETTm2_data(column='LULL')
    patch_len = args.patch_len  # 96
    seq_patch_r = 7
    pred_patch_r = 1
    pred_len = patch_len * pred_patch_r
    seq_len = patch_len * seq_patch_r

    split_ratio_list = np.linspace(1 / 2, 7 / 8, 60)  # ###

    split_idx_list = [int(data.shape[1] * r) for r in split_ratio_list]
    res_dir = os.path.join('./motivation', 'flip')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    result_dict_dict = {}  #

    flip_methods = [None, 'vertical']

    for flip_method in flip_methods:
        for split_idx in split_idx_list:
            seq = data[:, split_idx - seq_len:split_idx, :]
            # flip
            if flip_method is None:
                _seq = seq
            elif flip_method == 'vertical':
                _seq = seq * -1
            else:
                raise ValueError(f"flip_method={flip_method}")
            # inference
            _pred_total = exp.raw_inference(_seq.copy(), pred_patch_r)
            _pred = _pred_total[:, -pred_len:, :]
            # flip back
            if flip_method is None:
                pred_line = np.concatenate((seq[0, :, 0], _pred[0, :, 0]), axis=0)
            elif flip_method == 'vertical':
                pred_line = np.concatenate((seq[0, :, 0], -1 * (_pred[0, :, 0])), axis=0)
            else:
                raise ValueError(f"flip_method={flip_method}")
            truth_line = data[0, split_idx - seq_len:split_idx + pred_len, 0]
            assert len(pred_line) == len(truth_line), \
                f"len(pred_line)={len(pred_line)}, len(truth_line)={len(truth_line)}"
            mae, mse, rmse, mape, mspe = metric(pred_line[-pred_len:], truth_line[-pred_len:])
            result_dict_dict[f"flip_method_{flip_method}_split_idx_{split_idx}"] = \
                {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}
            print(f"flip_method={flip_method}, split_idx={split_idx}, "
                  f"mae={mae}, mse={mse}, rmse={rmse}, mape={mape}, mspe={mspe}")
        # 画个图吧
        plt.figure(figsize=(len(truth_line) // patch_len * 5, 5))
        plt.plot(pred_line, label="pred")
        plt.plot(truth_line, label="truth")
        plt.legend()
        plt.savefig(os.path.join(res_dir, f'flip_method_{flip_method}_split_idx_{split_idx}.pdf'), bbox_inches='tight')
    np.save(os.path.join(res_dir, f'result_dict_dict.npy'), result_dict_dict)
    # 画个图吧
    # Bar: x=flip_method y=mean_mse
    mean_mse_list = []
    for flip_method in flip_methods:
        mse_list = [result_dict_dict[f"flip_method_{flip_method}_split_idx_{split_idx}"]["mse"]
                    for split_idx in split_idx_list]
        mean_mse = np.mean(mse_list)
        mean_mse_list.append(mean_mse)
    plt.figure(figsize=(8, 6))
    flip_methods_names = [x if x is not None else 'None' for x in flip_methods]
    plt.bar(flip_methods_names, mean_mse_list, width=0.1)
    plt.xlabel("flip_method")
    plt.ylabel("mean_mse")
    plt.savefig(os.path.join(res_dir, f'mean_mse.pdf'), bbox_inches='tight')
    # 画个图吧
    # histigram：x=flip_method y=num_point
    # 对于不同的idx，找到mse小的排名靠前的flip_method，y[flip_method]+1
    rate_ratio = 0.5  # 一半
    y = [0] * len(flip_methods)
    for split_idx in split_idx_list:
        mse_list = []
        for flip_method in flip_methods:
            mse_list.append(result_dict_dict[f"flip_method_{flip_method}_split_idx_{split_idx}"]["mse"])
        mse_list = np.array(mse_list)
        mse_list = np.argsort(mse_list)
        mse_list = mse_list[:int(len(mse_list) * rate_ratio)]
        for flip_method in mse_list:
            y[flip_method] += 1
    plt.figure(figsize=(8, 6))
    plt.bar(flip_methods_names, y, width=0.1)
    plt.xlabel("flip_method")
    plt.ylabel("num_point")
    plt.savefig(os.path.join(res_dir, f'num_point.pdf'), bbox_inches='tight')


def moti_len():
    # 比较不同的seq_len的预测情况差别

    # data = get_ETTm2_data(column='HULL')
    data = get_ETTm2_data(column='LULL')
    patch_len = args.patch_len  # 96
    pred_patch_r = 1
    pred_len = patch_len * pred_patch_r

    split_ratio_list = np.linspace(1 / 2, 7 / 8, 60)  # ###
    split_idx_list = [int(data.shape[1] * r) for r in split_ratio_list]

    res_dir = os.path.join('./motivation', 'seq_len')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    result_dict_dict = {}  #

    seq_patch_ratios = np.arange(1, 30, 1)
    seq_lens = [patch_len * seq_patch_ratio for seq_patch_ratio in seq_patch_ratios]

    for seq_patch_ratio in seq_patch_ratios:
        seq_len = patch_len * seq_patch_ratio
        for split_idx in split_idx_list:
            seq = data[:, split_idx - seq_len:split_idx, :]
            # inference
            _pred_total = exp.raw_inference(seq.copy(), pred_patch_r)
            _pred = _pred_total[:, -pred_len:, :]
            pred_line = np.concatenate((seq[0, :, 0], _pred[0, :, 0]), axis=0)
            truth_line = data[0, split_idx - seq_len:split_idx + pred_len, 0]
            assert len(pred_line) == len(truth_line), \
                f"len(pred_line)={len(pred_line)}, len(truth_line)={len(truth_line)}"
            mae, mse, rmse, mape, mspe = metric(pred_line[-pred_len:], truth_line[-pred_len:])
            result_dict_dict[f"seq_len_{seq_len}_split_idx_{split_idx}"] = \
                {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}
            print(f"seq_len={seq_len}, split_idx={split_idx}, "
                  f"mae={mae}, mse={mse}, rmse={rmse}, mape={mape}, mspe={mspe}")
        # 画个图吧
        plt.figure(figsize=(len(truth_line) // patch_len * 5, 5))
        plt.plot(pred_line, label="pred")
        plt.plot(truth_line, label="truth")
        plt.legend()
        plt.savefig(os.path.join(res_dir, f'seq_len_{seq_len}_split_idx_{split_idx}.pdf'), bbox_inches='tight')
    np.save(os.path.join(res_dir, f'result_dict_dict.npy'), result_dict_dict)
    # 画个图吧
    # Bar: x=seq_len y=mean_mse
    mean_mse_list = []
    for seq_len in seq_lens:
        mse_list = [result_dict_dict[f"seq_len_{seq_len}_split_idx_{split_idx}"]["mse"]
                    for split_idx in split_idx_list]
        mean_mse = np.mean(mse_list)
        mean_mse_list.append(mean_mse)
    plt.figure(figsize=(8, 6))
    plt.bar(seq_lens, mean_mse_list)
    plt.xlabel("seq_len")
    plt.ylabel("mean_mse")
    plt.savefig(os.path.join(res_dir, f'mean_mse.pdf'), bbox_inches='tight')
    # 画个图吧
    # histigram：x=seq_len y=num_point
    # 对于不同的idx，找到mse小的排名靠前的seq_len，y[seq_len]+1
    rate_ratio = 0.25
    y = [0] * len(seq_lens)
    for split_idx in split_idx_list:
        mse_list = []
        for seq_len in seq_lens:
            mse_list.append(result_dict_dict[f"seq_len_{seq_len}_split_idx_{split_idx}"]["mse"])
        mse_list = np.array(mse_list)
        mse_list = np.argsort(mse_list)
        mse_list = mse_list[:int(len(mse_list) * rate_ratio)]
        for seq_len in mse_list:
            y[seq_len] += 1
    plt.figure(figsize=(8, 6))
    plt.bar(seq_lens, y)
    plt.xlabel("seq_len")
    plt.ylabel("num_point")
    plt.savefig(os.path.join(res_dir, f'num_point.pdf'), bbox_inches='tight')


def gen_sin_data():
    n = 30000
    x = np.linspace(0, 3000, n)
    y = np.sin(x)
    # 保存sin到csv
    np.savetxt("sin.csv", y, delimiter=",")
    # 画图展示一下
    plt.plot(y)
    # plt.show() if is_pycharm() else plt.savefig("sin.pdf")


def gen_sin2_data():
    # 在sin的基础上，保持频率不变，但修改振幅和初始相位
    # 期望观察最优的seq_l和sample_r不变！！！！！！！！！！！
    n = 30000
    x = np.linspace(0, 3000, n) + 1000
    y = np.sin(x)
    np.savetxt("sin2.csv", y, delimiter=",")


def gen_sin3_data():
    n = 30000
    x = np.linspace(0, 3000, n) + 1432
    y = np.sin(x)
    np.savetxt("sin3.csv", y, delimiter=",")


def gen_sin4_data():
    n = 30000
    x = np.linspace(0, 3000, n)
    y = np.sin(x) + 12
    np.savetxt("sin4.csv", y, delimiter=",")


def gen_sin5_data():
    n = 30000
    x = np.linspace(0, 3000, n)
    y = np.sin(x) - 2
    np.savetxt("sin5.csv", y, delimiter=",")


def gen_sin6_data():
    n = 30000
    x = np.linspace(0, 3000, n) * 2
    y = np.sin(x)
    np.savetxt("sin6.csv", y, delimiter=",")


def gen_sin7_data():
    n = 30000
    x = np.linspace(0, 3000, n) * (-0.3)
    y = np.sin(x)
    np.savetxt("sin7.csv", y, delimiter=",")


def gen_sin8_data():
    n = 30000
    x = np.linspace(0, 3000, n)
    y = np.sin(x) * 2
    np.savetxt("sin8.csv", y, delimiter=",")


def gen_sin9_data():
    n = 30000
    x = np.linspace(0, 3000, n)
    y = np.sin(x) * (-0.3)
    np.savetxt("sin9.csv", y, delimiter=",")


def gen_sin_hard_data():
    n = 30000
    x = np.linspace(0, 3000, n)
    y = np.sin(x) + np.sin(2 * x) + np.sin(3 * x)
    y = y
    np.savetxt("sin_hard.csv", y, delimiter=",")


def get_sin_data(name):
    # out:[1,n,1]
    file_path = name + '.csv'  # sin sin2 sin3
    data = np.genfromtxt(file_path, delimiter=",")
    data = np.expand_dims(data, 0)
    data = np.expand_dims(data, 2)
    return data


def moti_len_sample(sin_data_name):
    # 探究 seq_len 和 sample_r 对预测效果的复合影响
    data = get_sin_data(sin_data_name)
    print("data.shape", data.shape)

    patch_len = 96

    pred_patch_r = 3
    pred_len = patch_len * pred_patch_r

    split_ratio_list = np.linspace(13 / 16, 15 / 16, 30)
    split_idx_list = [int(data.shape[1] * r) for r in split_ratio_list]

    seq_lens = np.arange(1, 30, 3) * patch_len
    print("seq_lens", seq_lens)  # seq_lens [  96  576 1056 1536 2016 2496]
    sample_ratios = np.around(np.linspace(1, 10, 9 + 1), 4)  # sample_ratios [ 1.   2.8  4.6  6.4  8.2 10. ]
    print("sample_ratios", sample_ratios)

    res_dir = os.path.join('./motivation', 'seq_len_sample_r', sin_data_name)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    result_dict_dict = {}
    seq_lens_bar = tqdm(seq_lens, desc='Processing', ncols=100)  # 一次一分钟
    for seq_len in seq_lens_bar:
        for sample_r in sample_ratios:
            for split_idx in split_idx_list:
                # print("seq_len", seq_len, "sample_r", sample_r, "split_idx", split_idx)
                assert split_idx - seq_len >= 0
                seq = data[:, split_idx - seq_len:split_idx, :]
                # down _sample data -> model input -> model output -> up _sample output
                seq_patch_r = seq_len // patch_len
                model_seq_len = ceil(seq_patch_r / sample_r) * patch_len
                model_pred_patch_r = ceil(pred_patch_r / sample_r)
                model_pred_len = patch_len * model_pred_patch_r
                # ##################### begin
                _seq = seq[0, :, 0]
                _seq = signal.resample(_seq, ceil(seq_len / sample_r))  # ceil(seq_len / sample_r)重要!!!!!! 不是整数倍
                # 长度不够patch整数倍的话要在前面补0
                _seq = np.concatenate((np.zeros(patch_len - len(_seq) % patch_len), _seq)) \
                    if len(_seq) % patch_len != 0 else _seq
                assert len(_seq) == model_seq_len, f"len(_seq)={len(_seq)}, model_seq_len={model_seq_len}"
                _seq = _seq[np.newaxis, :, np.newaxis]
                assert _seq.shape[1] % patch_len == 0 and _seq.shape[2] == 1 and _seq.shape[0] == 1, \
                    f"_seq.shape={_seq.shape}"
                # print(_seq[0, :5, 0])
                _pred_total = exp.raw_inference(_seq.copy(), model_pred_patch_r)
                assert _pred_total.shape[1] == model_seq_len + model_pred_len, \
                    f"_pred_total.shape[1]={_pred_total.shape[1]}, " \
                    f"model_seq_len+model_pred_len={model_seq_len + model_pred_len}"
                _pred = _pred_total[:, -model_pred_len:, :]
                # print(_pred[0, :5, 0])
                _pred = signal.resample(_pred[0, :, 0], ceil(model_pred_len * sample_r))  # #### 当心有错！！！！scale相似的问题？
                assert pred_len <= len(_pred), f"pred_len={pred_len}, len(_pred)={len(_pred)}"
                # ##################### end
                pred = _pred[:pred_len].reshape(1, -1, 1)
                pred_line = np.concatenate((seq[0, :, 0], pred[0, :, 0]), axis=0)
                truth_line = data[0, split_idx - seq_len:split_idx + pred_len, 0]
                assert len(pred_line) == len(truth_line), \
                    f"len(pred_line)={len(pred_line)}, len(truth_line)={len(truth_line)}"
                mae, mse, rmse, mape, mspe = metric(pred_line[-pred_len:], truth_line[-pred_len:])
                result_dict_dict[f"seq_len_{seq_len}_sample_r_{sample_r}_split_idx_{split_idx}"] = \
                    {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}
                print(f"seq_len={seq_len}, sample_r={sample_r}, split_idx={split_idx}, "
                      f"mae={mae}, mse={mse}, rmse={rmse}, mape={mape}, mspe={mspe}")
            # 画个图吧
            plt.figure(figsize=(len(truth_line) // patch_len * 5, 5))
            plt.plot(pred_line, label="pred")
            plt.plot(truth_line, label="truth")
            plt.legend()
            plt.savefig(os.path.join(res_dir, f'seq_len_{seq_len}_sample_r_{sample_r}_split_idx_{split_idx}.pdf'),
                        bbox_inches='tight')

    np.save(os.path.join(res_dir, f'result_dict_dict.npy'), result_dict_dict)

    # 画个图吧
    # Bar：x=seq_len+sample_r y=mean_mse
    mean_mse_list = []
    std_mse_list = []
    for seq_len in seq_lens:
        for sample_r in sample_ratios:
            mse_list = [result_dict_dict[f"seq_len_{seq_len}_sample_r_{sample_r}_split_idx_{split_idx}"][
                            "mse"]
                        for split_idx in split_idx_list]
            mean_mse_list.append(np.mean(mse_list))
            std_mse_list.append(np.std(mse_list))
    plt.figure(figsize=(8, 6))
    names = [f"{seq_len}+{sample_r}" for seq_len in seq_lens for sample_r in sample_ratios]
    # plt.bar(names, mean_mse_list, width=0.1)
    plt.errorbar(names, mean_mse_list, yerr=std_mse_list, fmt='o', color='red', ecolor='green', elinewidth=2, capsize=4)
    plt.xlabel("seq_len+sample_r")
    plt.ylabel("mean_mse")
    plt.savefig(os.path.join(res_dir, f'mean_mse.pdf'), bbox_inches='tight')
    # 画个图吧
    # Bar：x=seq_len y=mean_mse
    mean_mse_list = []
    std_mse_list = []
    for seq_len in seq_lens:
        mse_list = [result_dict_dict[f"seq_len_{seq_len}_sample_r_{sample_r}_split_idx_{split_idx}"]["mse"]
                    for sample_r in sample_ratios for split_idx in split_idx_list]
        mean_mse_list.append(np.mean(mse_list))
        std_mse_list.append(np.std(mse_list))
    plt.figure(figsize=(8, 6))
    names = [f"{seq_len}" for seq_len in seq_lens]
    # plt.bar(names, mean_mse_list, width=0.1)
    plt.errorbar(names, mean_mse_list, yerr=std_mse_list, fmt='o', color='red', ecolor='green', elinewidth=2, capsize=4)
    plt.xlabel("seq_len")
    plt.ylabel("mean_mse")
    plt.savefig(os.path.join(res_dir, f'mean_mse_seq_len.pdf'), bbox_inches='tight')
    # 画个图吧
    # Bar：x=sample_r y=mean_mse
    mean_mse_list = []
    std_mse_list = []
    for sample_r in sample_ratios:
        mse_list = [result_dict_dict[f"seq_len_{seq_len}_sample_r_{sample_r}_split_idx_{split_idx}"]["mse"]
                    for seq_len in seq_lens for split_idx in split_idx_list]
        mean_mse_list.append(np.mean(mse_list))
        std_mse_list.append(np.std(mse_list))
    plt.figure(figsize=(8, 6))
    names = [f"{sample_r}" for sample_r in sample_ratios]
    # plt.bar(names, mean_mse_list, width=0.1)
    plt.errorbar(names, mean_mse_list, yerr=std_mse_list, fmt='o', color='red', ecolor='green', elinewidth=2, capsize=4)
    plt.xlabel("sample_r")
    plt.ylabel("mean_mse")
    plt.savefig(os.path.join(res_dir, f'mean_mse_sample_r.pdf'), bbox_inches='tight')

    # 画个图吧
    # histigram：x=seq_len:sample_r y=num_point
    # 对于不同的idx，找到mse小的排名靠前的seq_len+sample_r，y[seq_len+sample_r]+1
    rate_ratio = 0.25
    y = [0] * len(seq_lens) * len(sample_ratios)
    for split_idx in split_idx_list:
        mse_list = []
        for seq_len in seq_lens:
            for sample_r in sample_ratios:
                mse_list.append(result_dict_dict[f"seq_len_{seq_len}_sample_r_{sample_r}_split_idx_{split_idx}"]["mse"])
        mse_list = np.array(mse_list)
        idx_list = np.argsort(mse_list)
        idx_list = idx_list[:int(len(idx_list) * rate_ratio)]
        for idx in idx_list:
            seq_len_idx = idx // len(sample_ratios)
            sample_r_idx = idx % len(sample_ratios)
            y[seq_len_idx * len(sample_ratios) + sample_r_idx] = y[seq_len_idx * len(sample_ratios) + sample_r_idx] + 1
    plt.figure(figsize=(8, 6))
    names = [f"{seq_len}+{sample_r}" for seq_len in seq_lens for sample_r in sample_ratios]
    plt.bar(names, y, width=0.1)
    plt.xlabel("seq_len+sample_r")
    plt.ylabel("num_point")
    plt.savefig(os.path.join(res_dir, f'num_point.pdf'), bbox_inches='tight')
    # 画个图吧
    # histigram：x=seq_len y=num_point
    # 对于不同的idx，找到mse小的排名靠前的seq_len，y[seq_len]+1
    rate_ratio = 0.25
    y = [0] * len(seq_lens)
    for split_idx in split_idx_list:
        mse_list = []
        for seq_len in seq_lens:
            for sample_r in sample_ratios:
                mse_list.append(result_dict_dict[f"seq_len_{seq_len}_sample_r_{sample_r}_split_idx_{split_idx}"]["mse"])
        mse_list = np.array(mse_list)
        idx_list = np.argsort(mse_list)
        idx_list = idx_list[:int(len(idx_list) * rate_ratio)]
        for idx in idx_list:
            seq_len_idx = idx // len(sample_ratios)
            y[seq_len_idx] = y[seq_len_idx] + 1
    plt.figure(figsize=(8, 6))
    names = [f"{seq_len}" for seq_len in seq_lens]
    plt.bar(names, y, width=0.1)
    plt.xlabel("seq_len")
    plt.ylabel("num_point")
    plt.savefig(os.path.join(res_dir, f'num_point_seq_len.pdf'), bbox_inches='tight')
    # 画个图吧
    # histigram：x=sample_r y=num_point
    # 对于不同的idx，找到mse小的排名靠前的sample_r，y[sample_r]+1
    rate_ratio = 0.25
    y = [0] * len(sample_ratios)
    for split_idx in split_idx_list:
        mse_list = []
        for sample_r in sample_ratios:
            for seq_len in seq_lens:
                mse_list.append(result_dict_dict[f"seq_len_{seq_len}_sample_r_{sample_r}_split_idx_{split_idx}"]["mse"])
        mse_list = np.array(mse_list)
        idx_list = np.argsort(mse_list)
        idx_list = idx_list[:int(len(idx_list) * rate_ratio)]
        for idx in idx_list:
            sample_r_idx = idx // len(sample_ratios)
            y[sample_r_idx] = y[sample_r_idx] + 1
    plt.figure(figsize=(8, 6))
    names = [f"{sample_r}" for sample_r in sample_ratios]
    plt.bar(names, y, width=0.1)
    plt.xlabel("sample_r")
    plt.ylabel("num_point")
    plt.savefig(os.path.join(res_dir, f'num_point_sample_r.pdf'), bbox_inches='tight')

    # 平行坐标图！
    _seq_len_list = []
    _sample_ratio_list = []
    mean_mse_list = []
    for seq_len in seq_lens:
        for sample_r in sample_ratios:
            mse_list = [result_dict_dict[f"seq_len_{seq_len}_sample_r_{sample_r}_split_idx_{split_idx}"]["mse"]
                        for split_idx in split_idx_list]
            mean_mse = np.mean(mse_list)
            mean_mse_list.append(mean_mse)
            _seq_len_list.append(seq_len)
            _sample_ratio_list.append(sample_r)
    data = [go.Parcoords(
        line=dict(color='blue'),
        dimensions=list([
            dict(range=[min(_seq_len_list), max(_seq_len_list)],
                 label='seq_len', values=_seq_len_list),
            dict(range=[min(_sample_ratio_list), max(_sample_ratio_list)],
                 label='sample_r', values=_sample_ratio_list),
            dict(range=[min(mean_mse_list), max(mean_mse_list)],
                 label='mean_mse', values=mean_mse_list)
        ])
    )]
    layout = go.Layout(title="My first parallel coordinates")
    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig, filename=os.path.join(res_dir, 'mean_mse_parcoords.html'))


if __name__ == '__main__':
    gen_sin_hard_data()
    for sin_data_name in ['sin_hard']:
        moti_len_sample(sin_data_name)
