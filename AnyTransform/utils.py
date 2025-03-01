import itertools
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import plotly.graph_objs as go
from matplotlib import pyplot as plt


# from AnyTransform.pipeline import get_params_space_and_org

def get_params_space_and_org(fast_mode=None):
    if fast_mode is None:
        from parser import fast_mode,  ablation
        patch_len=96
    else:
        patch_len = 96
        ablation = 'none'
    params_space = {
        'sampler_factor': {
            'type': 'int',
            'values': np.arange(1, 2 + 1, 1)  # 减小space
            # 'values': np.arange(1, 3 + 1, 1)  # TODO：未来再改成float # 现在不怎么过拟合了，有必要扩大吗？效果一般
            # if ablation != 'Sampler' else [1]
            # 'type': 'float',
            # 'values': np.linspace(1, 2, 5)  # array([1.  , 1.25, 1.5 , 1.75, 2.  ]) # 连续取值太多
            # 'type': 'str',
            # 'values': [0.5, 1]
        },
        'trimmer_seq_len': {
            'type': 'int',  # (40,96,1) Chronos+Cuda稳定报错
            'values': np.arange(5 * patch_len, (15 + 1) * patch_len, int(patch_len * 1))  # 补align危险？？？
            # if ablation != 'Trimmer' else [7 * patch_len],
        },
        'aligner_mode': {
            'type': 'str',
            'values': ['none', 'data_patch']  # 减小Space
            # 'values': ['none', 'data_patch', 'model_patch']  # hope2
            # if ablation != 'Aligner' else ['none'],
        },
        'aligner_method': {
            'type': 'str',  # zero_pad会被明显排除 不过会影响整体可视化和debug
            'values': ['edge_pad']  # 减小Space
            # 'values': ['mean_pad', 'edge_pad']  # none+model_patch变差？？？确实 特别是UTSD。。。trim更差。。。
            # if ablation != 'Aligner' else ['edge_pad'],  # 跟org一致
            # 'mean_pad', 'edge_pad', 'none', 'trim'
        },
        'normalizer_method': {
            'type': 'str',  # 'minmax', 'maxabs' 减小搜索空间 ????????????'minmax', 'maxabs'???????????
            'values': ['none', 'standard', 'robust']  # robust略慢...！！
            # if ablation != 'Normalizer' else ['none'],
            # 'values': ['none', 'standard', 'robust', 'minmax', 'maxabs']
        },
        'normalizer_mode': {
            'type': 'str',  # train overfit??? ???????????????????????leak！ 'history'导致Weather坏?????
            'values': ['input']
            # if ablation != 'Normalizer' else ['input'],
            # 'values': ['none', 'input', 'history', 'train'] 'dataset'
        },
        'normalizer_ratio': {  # new!!!
            'type': 'str',
            'values': [1]  # 减小space
            # 'values': [1, 0.5, 0.25]
            # if ablation != 'Normalizer' else [1],
        },
        'inputer_detect_method': {
            'type': 'str',  # iqr计算时间长了一点(1/2 model)！！
            # * Original value
            # 'values': ['none', '3_sigma', '1.5_iqr']
            # * Remove inputer
            'values': ['none']
            # if ablation != 'Inputer' else ['none'],
            # 'values': ['none', '3_sigma', '1.5_iqr']
        },
        'inputer_fill_method': {  # forward_fill感觉很多时候也不如不impute... # forward_fill在ETT上差！
            'type': 'str',  # 'forward_fill', 'backward_fill' 减小搜索空间 # rolling_mean有时候遇大倾斜很坏！！！
            'values': ['linear_interpolate']
            # if ablation != 'Inputer' else ['linear_interpolate'],
            # 'values': ['none', 'linear_interpolate', 'rolling_mean', 'forward_fill']
        },
        'warper_method': {  # 'boxcox'+Uni2ts -> 有时候nan 而且秒级 # yeojohnson 有时候会-1847倍...overfit..??
            'type': 'str',  # 'log' 'sqrt' 坏
            'values': ['none', 'log']  # 减小space
            # 'values': ['none', 'log', 'sqrt']  # ！！！！log能让Electricity和Weather变好！->本质：变得不那么差; 但是整体会'坏...
            # if ablation != 'Warper' else ['none'],
            # 'values': ['none', 'log', 'boxcox', 'yeojohnson']
        },
        'decomposer_period': {  # 有点太慢了...并行抢cpu # 现在还行 # 整体貌似会变差？rand？
            'type': 'str',
            'values': ['none']
            # if ablation != 'Decomposer' else ['none'],
            # 'values': ['none', '60', '24', '30', '7', '12', '365']
            # '60', '24', '365'？
            # 12 6 4 24
            # '24', '60'
        },
        'decomposer_components': {
            'type': 'str',  # 'trend+season', 'season+residual', 'trend+residual'
            'values': ['none']
            # if ablation != 'Decomposer' else ['none'],
            # 'values': ['trend', 'season', 'residual', 'trend+season', 'season+residual', 'trend+residual', 'none']
            # 'trend', 'season', 'trend+season', 'trend+residual', 'season+residual'
            # 'trend+season', 'trend+residual', 'season+residual'
            # 'season+residual', 'season'
            # 'season+residual', 'season'
        },
        # Differentiator
        'differentiator_n': {
            'type': 'int',
            # * Original value
            # 'values': [0, 1]
            # * Remove Differentiator
            'values': [0]
            # if ablation != 'Differentiator' else [0],
        },
        'pipeline_name': {
            'type': 'str',
            'values': ['infer1', 'infer3']  # 减小Space
            # 'values': ['infer1', 'infer2', 'infer3']
            # if ablation != 'Pipeline' else ['infer1'],  # 跟org一致
        },
        'denoiser_method': {
            'type': 'str',  # 'moving_average' 配上 forward—fill导致UTSD在ETT上很差！？ # 'fft'没用？
            'values': ['none', 'ewma']  # 减小Space
            # 'values': ['none', 'ewma', 'moving_average']
            # if ablation != 'Denoiser' else ['none'],
        },
        'clip_factor': {
            'type': 'str',
            'values': ['none', '0', '0.25']
            # if ablation != 'Clipper' else ['none'],
        }
        # 大约十万的space yes
    }
    # logging.info(f"params_space={params_space}")
    # 注意：这个setting其实是比较随意的，最终只是为了说明默认值的效果一般（最好follow论文的设置）
    origin_param_dict = {  # FIXME：
        'sampler_factor': 1,
        'trimmer_seq_len': patch_len * 7,
        'aligner_mode': 'none',
        'aligner_method': 'edge_pad',  # model_patch之后org一定需要是none！！！
        'normalizer_method': 'none',  # FIXME： 目前已经使用了Timer内置的std的scaler
        'normalizer_mode': 'input',
        'normalizer_ratio': 1,
        'inputer_detect_method': 'none',
        'inputer_fill_method': 'linear_interpolate',
        'warper_method': 'none',
        'decomposer_period': 'none',
        'decomposer_components': 'none',
        'differentiator_n': 0,
        'pipeline_name': 'infer3',
        'denoiser_method': 'none',
        'clip_factor': 'none'  # FIXME: 原来是0 但实际上不影响，只是算子内部的clip
    }
    # logging.info(f"origin_param_dict={origin_param_dict}")
    return params_space, origin_param_dict

def get_params_space_and_org_cov(fast_mode=None):
    if fast_mode is None:
        from parser import fast_mode,  ablation
        patch_len=96
    else:
        patch_len = 96
        ablation = 'none'
    params_space = {
        'sampler_factor': {
            'type': 'int',
            'values': np.arange(1, 2 + 1, 1)  # 减小space
            # 'values': np.arange(1, 3 + 1, 1)  # TODO：未来再改成float # 现在不怎么过拟合了，有必要扩大吗？效果一般
            # if ablation != 'Sampler' else [1]
            # 'type': 'float',
            # 'values': np.linspace(1, 2, 5)  # array([1.  , 1.25, 1.5 , 1.75, 2.  ]) # 连续取值太多
            # 'type': 'str',
            # 'values': [0.5, 1]
        },
        'trimmer_seq_len': {
            'type': 'int',  # (40,96,1) Chronos+Cuda稳定报错
            'values': np.arange(5 * patch_len, (15 + 1) * patch_len, int(patch_len * 1))  # 补align危险？？？
            # if ablation != 'Trimmer' else [7 * patch_len],
        },
        'aligner_mode': {
            'type': 'str',
            'values': ['none', 'data_patch']  # 减小Space
            # 'values': ['none', 'data_patch', 'model_patch']  # hope2
            # if ablation != 'Aligner' else ['none'],
        },
        'aligner_method': {
            'type': 'str',  # zero_pad会被明显排除 不过会影响整体可视化和debug
            'values': ['edge_pad']  # 减小Space
            # 'values': ['mean_pad', 'edge_pad']  # none+model_patch变差？？？确实 特别是UTSD。。。trim更差。。。
            # if ablation != 'Aligner' else ['edge_pad'],  # 跟org一致
            # 'mean_pad', 'edge_pad', 'none', 'trim'
        },
        'normalizer_method': {
            'type': 'str',  # 'minmax', 'maxabs' 减小搜索空间 ????????????'minmax', 'maxabs'???????????
            'values': ['none', 'standard', 'robust']  # robust略慢...！！
            # if ablation != 'Normalizer' else ['none'],
            # 'values': ['none', 'standard', 'robust', 'minmax', 'maxabs']
        },
        'normalizer_mode': {
            'type': 'str',  # train overfit??? ???????????????????????leak！ 'history'导致Weather坏?????
            'values': ['input']
            # if ablation != 'Normalizer' else ['input'],
            # 'values': ['none', 'input', 'history', 'train'] 'dataset'
        },
        'normalizer_ratio': {  # new!!!
            'type': 'str',
            'values': [1]  # 减小space
            # 'values': [1, 0.5, 0.25]
            # if ablation != 'Normalizer' else [1],
        },
        'inputer_detect_method': {
            'type': 'str',  # iqr计算时间长了一点(1/2 model)！！
            'values': ['none', '3_sigma', '1.5_iqr']
            # if ablation != 'Inputer' else ['none'],
            # 'values': ['none', '3_sigma', '1.5_iqr']
        },
        # ! Covariate
        'inputer_detect_method_cov': {
            'type': 'str',  # iqr计算时间长了一点(1/2 model)！！
            'values': ['none', '3_sigma', '1.5_iqr']
            # if ablation != 'Inputer' else ['none'],
            # 'values': ['none', '3_sigma', '1.5_iqr']
        },
        'inputer_fill_method': {  # forward_fill感觉很多时候也不如不impute... # forward_fill在ETT上差！
            'type': 'str',  # 'forward_fill', 'backward_fill' 减小搜索空间 # rolling_mean有时候遇大倾斜很坏！！！
            'values': ['linear_interpolate']
            # if ablation != 'Inputer' else ['linear_interpolate'],
            # 'values': ['none', 'linear_interpolate', 'rolling_mean', 'forward_fill']
        },
        'warper_method': {  # 'boxcox'+Uni2ts -> 有时候nan 而且秒级 # yeojohnson 有时候会-1847倍...overfit..??
            'type': 'str',  # 'log' 'sqrt' 坏
            'values': ['none', 'log']  # 减小space
            # 'values': ['none', 'log', 'sqrt']  # ！！！！log能让Electricity和Weather变好！->本质：变得不那么差; 但是整体会'坏...
            # if ablation != 'Warper' else ['none'],
            # 'values': ['none', 'log', 'boxcox', 'yeojohnson']
        },
        'decomposer_period': {  # 有点太慢了...并行抢cpu # 现在还行 # 整体貌似会变差？rand？
            'type': 'str',
            'values': ['none']
            # if ablation != 'Decomposer' else ['none'],
            # 'values': ['none', '60', '24', '30', '7', '12', '365']
            # '60', '24', '365'？
            # 12 6 4 24
            # '24', '60'
        },
        'decomposer_components': {
            'type': 'str',  # 'trend+season', 'season+residual', 'trend+residual'
            'values': ['none']
            # if ablation != 'Decomposer' else ['none'],
            # 'values': ['trend', 'season', 'residual', 'trend+season', 'season+residual', 'trend+residual', 'none']
            # 'trend', 'season', 'trend+season', 'trend+residual', 'season+residual'
            # 'trend+season', 'trend+residual', 'season+residual'
            # 'season+residual', 'season'
            # 'season+residual', 'season'
        },
        # Differentiator
        'differentiator_n': {
            'type': 'int',
            'values': [0, 1]
            # if ablation != 'Differentiator' else [0],
        },
        # ! Covariate
        'differentiator_n_cov': {
            'type': 'int',
            'values': [0, 1]
            # if ablation != 'Differentiator' else [0],
        },
        'pipeline_name': {
            'type': 'str',
            'values': ['infer1', 'infer3']  # 减小Space
            # 'values': ['infer1', 'infer2', 'infer3']
            # if ablation != 'Pipeline' else ['infer1'],  # 跟org一致
        },
        'denoiser_method': {
            'type': 'str',  # 'moving_average' 配上 forward—fill导致UTSD在ETT上很差！？ # 'fft'没用？
            'values': ['none', 'ewma']  # 减小Space
            # 'values': ['none', 'ewma', 'moving_average']
            # if ablation != 'Denoiser' else ['none'],
        },
        # ! Covariate
        'denoiser_method_cov': {
            'type': 'str',  # 'moving_average' 配上 forward—fill导致UTSD在ETT上很差！？ # 'fft'没用？
            'values': ['none', 'ewma']  # 减小Space
            # 'values': ['none', 'ewma', 'moving_average']
            # if ablation != 'Denoiser' else ['none'],
        },
        'clip_factor': {
            'type': 'str',
            'values': ['none', '0', '0.25']
            # if ablation != 'Clipper' else ['none'],
        }
        # 大约十万的space yes
    }
    # logging.info(f"params_space={params_space}")
    # 注意：这个setting其实是比较随意的，最终只是为了说明默认值的效果一般（最好follow论文的设置）
    origin_param_dict = {  # FIXME：
        'sampler_factor': 1,
        'trimmer_seq_len': patch_len * 7,
        'aligner_mode': 'none',
        'aligner_method': 'edge_pad',  # model_patch之后org一定需要是none！！！
        'normalizer_method': 'none',  # FIXME： 目前已经使用了Timer内置的std的scaler
        'normalizer_mode': 'input',
        'normalizer_ratio': 1,
        'inputer_detect_method': 'none',
        'inputer_detect_method_cov': 'none',
        'inputer_fill_method': 'linear_interpolate',
        'warper_method': 'none',
        'decomposer_period': 'none',
        'decomposer_components': 'none',
        'differentiator_n': 0,
        'differentiator_n_cov': 0,
        'pipeline_name': 'infer3',
        'denoiser_method': 'none',
        'denoiser_method_cov': 'none',
        'clip_factor': 'none'  # FIXME: 原来是0 但实际上不影响，只是算子内部的clip
    }
    # logging.info(f"origin_param_dict={origin_param_dict}")
    return params_space, origin_param_dict

class TimeRecorder:
    def __init__(self, event_name):
        self.last_start_time = None
        self.duration = 0
        self.event_name = event_name

    def time_start(self):
        self.last_start_time = time.time()

    def time_end(self):
        d = time.time() - self.last_start_time
        self.duration += d
        logging.info(f"{self.event_name} last duration: {d}, cur total duration: {self.duration}")

    def get_total_duration(self):
        logging.info(f"Total {self.event_name} time: {self.duration}")
        return self.duration


def time_start():
    return time.time()


def log_time_delta(t, event_name):
    d = time.time() - t
    logging.info(f"{event_name} time: {d}")


def set_seed(seed):
    logging.info(f"Set seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Make the computations deterministic on GPU (if applicable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # scipy.random.seed(seed)
    # sklearn.utils.check_random_state(seed)
    # statsmodels.tools.check_random_state(seed)


def make_param_dict_unique(param_dict_list):
    unique_param_dict_list = []
    for param_dict in param_dict_list:
        if param_dict not in unique_param_dict_list:
            unique_param_dict_list.append(param_dict)
    logging.info(f"Unique param_dict_list: {unique_param_dict_list}")
    return unique_param_dict_list


def get_max_batch_size_for_cuda(model_name):
    # batch_size -> CUDA OOM 或者一些model相关的奇怪的报错
    # 模型越大下内存占用越多，但是可能patch_size变大，继而自回归占用显存变小
    # 推理也可能存在数值不稳定的问题->Uni2ts的logit报错 （amp
    if 'Chronos-tiny' in model_name:
        res = 600
    elif 'Timer' in model_name:
        res = 600
    elif 'MOIRAI-small' in model_name:
        res = 5000  # 6000
    elif 'MOIRAI-base' in model_name:
        res = 5000
    elif 'MOIRAI-large' in model_name:
        res = 4000
    elif 'Arima' in model_name:
        res = 1
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    # return min(res, maximum)  # for train speed
    return res


def my_clip(seq_in, seq_out, nan_inf_clip_factor=None, min_max_clip_factor=None):
    # nan_inf_clip_factor=3, min_max_clip_factor=2 ...
    # mean+1.5IQR-> max+0.25range
    if isinstance(seq_in, np.ndarray):
        max_values = np.max(seq_in, axis=1, keepdims=True)
        min_values = np.min(seq_in, axis=1, keepdims=True)
    elif isinstance(seq_in, torch.Tensor):
        max_values = torch.max(seq_in, dim=1, keepdim=True).values
        min_values = torch.min(seq_in, dim=1, keepdim=True).values
    else:
        raise ValueError(f"Unknown type: {type(seq_in)}")
    range_values = max_values - min_values

    assert nan_inf_clip_factor is not None or min_max_clip_factor is not None, \
        "nan_inf_clip_factor and min_max_clip_factor cannot be both None!"

   
    if isinstance(seq_out, np.ndarray):
        if nan_inf_clip_factor is not None and (np.isnan(seq_out).any() or np.isinf(seq_out).any()):
            max_allowed = max_values + nan_inf_clip_factor * range_values
            min_allowed = min_values - nan_inf_clip_factor * range_values
            logging.info(f"seq_out contains NaN values!!! \n")
            logging.debug(f"seq_out contains NaN values!!!: {seq_out}")
            # seq_out = np.nan_to_num(seq_out, nan=(max_values + min_values) / 2, posinf=max_allowed, neginf=min_allowed)
            seq_out = np.nan_to_num(seq_out, nan=max_allowed, posinf=max_allowed, neginf=min_allowed)  # nan hard punish
    elif isinstance(seq_out, torch.Tensor):
        if nan_inf_clip_factor is not None and (torch.isnan(seq_out).any() or torch.isinf(seq_out).any()):
            max_allowed = max_values + nan_inf_clip_factor * range_values
            min_allowed = min_values - nan_inf_clip_factor * range_values
            logging.info(f"seq_out contains NaN values!!! \n")
            logging.debug(f"seq_out contains NaN values!!!: {seq_out}")
            # seq_out = np.nan_to_num(seq_out, nan=(max_values + min_values) / 2, posinf=max_allowed, neginf=min_allowed)
            seq_out = torch.nan_to_num(seq_out, nan=max_allowed, posinf=max_allowed, neginf=min_allowed)  # nan hard punish
        
        # logging.warning(f"seq_out after filling NaN values: {seq_out}")
    if min_max_clip_factor is not None:
        max_allowed = max_values + min_max_clip_factor * range_values
        min_allowed = min_values - min_max_clip_factor * range_values
        seq_in_last_values = seq_in[:, -1:, :]
        if (seq_out > max_allowed).any() or (seq_out < min_allowed).any():
            logging.info(f"seq_out out of range!!!: \n")
            logging.debug(f"seq_out out of range!!!: {seq_out}")
            # seq_out = np.clip(seq_out, min_allowed, max_allowed)
            # logging.warning(f"seq_out after cl
            # ipping: {seq_out}")
            # FIXME：助长scale的气焰....危险 # 使用allowed！Ok ->对scale的处理还是有点问题？weizhi
            seq_out = smart_clip(seq_out, min_allowed, max_allowed, seq_in_last_values)
            # logging.warning(f"seq_out after smart scaling: {seq_out}")
    return seq_out


def smart_clip(seq, min_allowed, max_allowed, seq_in_last_values):
    assert len(seq.shape) == 3, "Input sequence must be 3D: (batch, time, feature)"
    assert min_allowed.shape == max_allowed.shape == (seq.shape[0], 1, seq.shape[2]), \
        "Min and max must have shape (batch, 1, feature)"

    batch, time, feature = seq.shape
    first_elements = seq[:, 0:1, :]  # Preserve the first elements
    # assert np.all(first_elements < max_values) and np.all(first_elements > min_values), \
    #     f"The first elements must be within min and max values: \n" \
    #     f"first_elements:{first_elements}, \nmin_values:{min_values}, \nmax_values:{max_values}"
    if isinstance(seq, np.ndarray):
        if np.any(first_elements > max_allowed) or np.any(first_elements < min_allowed):
            logging.info(f"The first elements must be within min and max allowed!!!\n")
            logging.debug(f"The first elements must be within min and max allowed: \n"
                          f"first_elements:{first_elements}, \nmin_allowed:{min_allowed}, \nmax_allowed:{max_allowed}")
            # return np.clip(seq, min_allowed, max_allowed)
            # FIXME：平移first使得跟last一样,即在(first,last)之间进行scale
            seq = seq - first_elements + seq_in_last_values
        seq_max_values = np.max(seq, axis=1, keepdims=True)  # Include the first element
        seq_min_values = np.min(seq, axis=1, keepdims=True)  # Include the first element isinstance(seq_in, torch.Tensor):
    elif isinstance(seq, torch.Tensor):
        if torch.any(first_elements > max_allowed) or torch.any(first_elements < min_allowed):
            logging.info(f"The first elements must be within min and max allowed!!!\n")
            logging.debug(f"The first elements must be within min and max allowed: \n"
                          f"first_elements:{first_elements}, \nmin_allowed:{min_allowed}, \nmax_allowed:{max_allowed}")
            # return torch.clamp(seq, min=min_allowed, max=max_allowed)
            # FIXME：平移first使得跟last一样,即在(first,last)之间进行scale
            seq = seq - first_elements + seq_in_last_values
        seq_max_values = torch.max(seq, dim=1, keepdim=True).values
        seq_min_values = torch.min(seq, dim=1, keepdim=True).values
    else:
        raise ValueError(f"Unknown type: {type(seq_in)}")
    

    # Apply scaling to the sequences that exceed the max values
    for i in range(batch):
        for j in range(feature):
            # 如果存在大于max的值，则把值介于(first,max)的数值进行scale
            tmp_seq = seq[i, :, j]
            first_value = first_elements[i, 0, j]
            max_value = max_allowed[i, 0, j]
            min_value = min_allowed[i, 0, j]
            seq_max_value = seq_max_values[i, 0, j]
            seq_min_value = seq_min_values[i, 0, j]
            if seq_max_value > max_value:
                scale = (max_value - first_value) / (seq_max_value - first_value)
                upper_mask = tmp_seq > first_value
                seq[i, upper_mask, j] = first_value + (seq[i, upper_mask, j] - first_value) * scale
            if seq_min_value < min_value:
                scale = (min_value - first_value) / (seq_min_value - first_value)
                lower_mask = tmp_seq < first_value
                seq[i, lower_mask, j] = first_value + (seq[i, lower_mask, j] - first_value) * scale
    return seq


def get_valid_params_mode_data(pd_data, mode, params_space):
    logging.info(f"Begin to get valid params mode data in mode={mode}...")
    mode_data = pd_data[pd_data['mode'] == mode]
    data_by_params = mode_data.groupby(list(params_space.keys())).size().reset_index(name='counts')
    max_step = data_by_params['counts'].max()
    logging.debug(f"max_step={max_step}")
    valid_params = data_by_params[data_by_params['counts'] == max_step].drop(columns='counts')
    valid_params_mode_data = mode_data.merge(valid_params, on=list(params_space.keys()))
    return valid_params_mode_data


def find_top_param_dict_list(pd_data, mode, params_space, top_k, metric_weight_dict, res_dir):
    logging.info(f"Begin to find top-{top_k} param dict list in mode={mode}...")
    metric_names = list(metric_weight_dict.keys())
    rank_metric_names = [f'{metric}_rank' for metric in metric_names]

    # Filter the dataframe to include only mode data and un-pruned params
    valid_params_mode_data = get_valid_params_mode_data(pd_data, mode, params_space)

    data_by_valid_params = valid_params_mode_data.groupby(list(params_space.keys()))
    mean_metrics_by_valid_params = data_by_valid_params[metric_names].mean().reset_index()

    # 计算加权平均值
    # Calculate ranks for each metric and add to DataFrame
    for metric in metric_weight_dict.keys():
        mean_metrics_by_valid_params[f'{metric}_rank'] = mean_metrics_by_valid_params[metric].rank()
        mean_metrics_by_valid_params[f'{metric}_rank'] *= metric_weight_dict.get(metric, 1)
    # Calculate mean rank across all metrics
    total_weight = sum(metric_weight_dict.values())
    mean_metrics_by_valid_params['final_rank'] = \
        mean_metrics_by_valid_params[rank_metric_names].sum(axis=1) / total_weight

    # save the mean_metrics_by_valid_params
    mean_metrics_by_valid_params.to_csv(os.path.join(res_dir, f'_{mode}_mean_metrics_by_params.csv'), index=False)

    # Sort parameter combinations based on mean rank
    sorted_pd_data = mean_metrics_by_valid_params.sort_values(by='final_rank', ascending=True) \
        .head(top_k if top_k != 'all' else len(mean_metrics_by_valid_params))

    # Print the sorted data for inspection
    logging.debug(f"sorted_pd_data=\n{sorted_pd_data.to_string()}")

    # Convert the sorted dataframe to a list of parameter dictionaries
    best_param_dict_list = sorted_pd_data.drop(columns=metric_names + ['final_rank'] + rank_metric_names) \
        .to_dict(orient='records')
    return best_param_dict_list


def find_top_param_dict_list_pareto(pd_data, mode, params_space, top_k, metric_weight_dict, metrics_for_pareto_must,
                                    multi_pareto_mode, res_dir):
    logging.info(f"Begin to find top-{top_k} param dict list in mode={mode} for multiple metric combinations...")
    metric_names = list(metric_weight_dict.keys())

    # Filter the dataframe to include only mode data and un-pruned params
    valid_params_mode_data = get_valid_params_mode_data(pd_data, mode, params_space)
    # Compute mean MSE and other metrics for each parameter combination
    data_by_valid_params = valid_params_mode_data.groupby(list(params_space.keys()))
    mean_metrics_by_params = data_by_valid_params[metric_names].mean().reset_index()

    # mean_metrics_by_params = mean_metrics_by_params.copy()
    for metric in metric_names:
        mean_metrics_by_params[f'{metric}_rank'] = mean_metrics_by_params[metric].rank()
        mean_metrics_by_params[f'{metric}_rank'] *= metric_weight_dict.get(metric, 1)

    if multi_pareto_mode is True:  # 多个指标的排列组合 形成多个帕里托前沿
        # Generate all possible metric combinations
        metric_names_list = [list(combo) for length in range(1, len(metric_names) + 1) for combo in
                             itertools.combinations(metric_names, length)]
        rm_idxes = []
        for idx, cur_metric_names in enumerate(metric_names_list):
            # must要求必须包含的所有must指标 或者 本身是must指标的子集合
            if set(metrics_for_pareto_must).issubset(cur_metric_names):
                continue
            elif set(cur_metric_names).issubset(metrics_for_pareto_must):
                continue
            else:
                rm_idxes.append(idx)
        for idx in rm_idxes[::-1]:
            metric_names_list.pop(idx)
    else:
        metric_names_list = [metrics_for_pareto_must]  # !!!

    # Store all results
    all_best_param_dict_list = []
    for cur_metric_names in metric_names_list:
        logging.info(f"cur_metric_names={cur_metric_names}")
        # Calculate mean rank across all metrics
        cur_metric_rank_names = [f'{metric}_rank' for metric in cur_metric_names]
        total_weight = sum(metric_weight_dict[metric] for metric in cur_metric_names)
        mean_metrics_by_params['final_rank'] = mean_metrics_by_params[cur_metric_rank_names] \
                                                   .sum(axis=1) / total_weight

        # if last then save
        if cur_metric_names == metric_names_list[-1]:  # 期望比较全面的保存
            mean_metrics_by_params.to_csv(os.path.join(res_dir, f'_{mode}_mean_metrics_by_params.csv'), index=False)

        # Sort parameter combinations based on mean rank
        sorted_pd_data = mean_metrics_by_params.sort_values(by='final_rank', ascending=True) \
            .head(top_k if top_k != 'all' else len(mean_metrics_by_params))
        # Convert the sorted dataframe to a list of parameter dictionaries
        best_param_dict_list = sorted_pd_data[list(params_space.keys())].to_dict(orient='records')
        all_best_param_dict_list.extend(best_param_dict_list)

    logging.info(f"len(all_best_param_dict_list)={len(all_best_param_dict_list)}")
    return all_best_param_dict_list


def calc_statistics(valid_params_mode_data, metric_names, our_param_dict):
    logging.info(f"Begin to calculate statistics ...")

    # Construct mask for the val_top1 param dict
    param_mask = np.logical_and.reduce([valid_params_mode_data[key] == value for key, value in our_param_dict.items()])

    res = {}
    # Get the metric values for the filtered data
    for metric_name in metric_names:
        logging.info(f"Calculating statistics for {metric_name}...")

        val_top1_metric_values = valid_params_mode_data[param_mask][metric_name].values

        # Calculate the mean, median, std, and IQR for val_top1 metric values
        val_top1_mean = np.mean(val_top1_metric_values)
        val_top1_median = np.median(val_top1_metric_values)
        val_top1_std = np.std(val_top1_metric_values)
        val_top1_q1 = np.percentile(val_top1_metric_values, 25)
        val_top1_q3 = np.percentile(val_top1_metric_values, 75)
        val_top1_iqr = val_top1_q3 - val_top1_q1
        val_top1_max = np.max(val_top1_metric_values)
        val_top1_min = np.min(val_top1_metric_values)

        res[f"{metric_name}_mean"] = val_top1_mean
        res[f"{metric_name}_median"] = val_top1_median
        res[f"{metric_name}_std"] = val_top1_std
        res[f"{metric_name}_iqr"] = val_top1_iqr
        res[f"{metric_name}_max"] = val_top1_max
        res[f"{metric_name}_min"] = val_top1_min
    return res


def find_top_param_dict_list_by_statistics(pd_data, mode, params_space, top_k, metric_weight_dict, stats_weight_dict,
                                           res_dir):
    logging.info(f"Begin to find top-{top_k} param dict list by statistics in mode={mode}...")

    metric_names = list(metric_weight_dict.keys())
    statistics_names = list(stats_weight_dict.keys())
    statistics_metric_names = [f'{metric}_{stat}' for metric in metric_names for stat in statistics_names]
    rank_metric_names = [f'{stat_metric}_rank' for stat_metric in statistics_metric_names]

    # Filter the dataframe to include only mode data and un-pruned params
    valid_params_mode_data = get_valid_params_mode_data(pd_data, mode, params_space)
    # FIXME：计算出不重复的valid_params_dict_list(先用group)
    valid_params_dict_list = valid_params_mode_data[list(params_space.keys())].drop_duplicates() \
        .to_dict(orient='records')
    logging.info(f"len(valid_params_dict_list)={len(valid_params_dict_list)}")
    logging.info(f"valid_params_mode_data=\n{valid_params_mode_data}")

    statistics_metrics_df = pd.DataFrame(columns=list(params_space.keys()) + statistics_metric_names)
    logging.info(f"Begin to calculate statistics for each parameter combination...")
    for param_dict in valid_params_dict_list:
        logging.info(f"Calculating statistics for param_dict={param_dict}...")
        statistics_dict = calc_statistics(valid_params_mode_data, metric_names, param_dict)
        logging.info(f"statistics_dict={statistics_dict}")
        # statistics_metrics_df = statistics_metrics_df.append({**param_dict, **statistics_dict}, ignore_index=True)
        statistics_metrics_df.loc[len(statistics_metrics_df)] = {**param_dict, **statistics_dict}

    logging.info(f"Begin to calculate ranks for each statistics metric...")
    for stat_metric in statistics_metric_names:
        stat = stat_metric.split('_')[-1]
        metric = '_'.join(stat_metric.split('_')[:-1])
        weight = metric_weight_dict[metric] * stats_weight_dict[stat]
        # 所有的都是越小越好
        statistics_metrics_df[f'{stat_metric}_rank'] = statistics_metrics_df[stat_metric].rank(ascending=True)
        statistics_metrics_df[f'{stat_metric}_rank'] *= weight

    total_weight = sum(metric_weight_dict[metric] * stats_weight_dict[stat]
                       for metric in metric_names for stat in statistics_names)
    statistics_metrics_df['final_rank'] = statistics_metrics_df[rank_metric_names].sum(axis=1) / total_weight

    statistics_metrics_df.to_csv(os.path.join(res_dir, f'_{mode}_statistics_metrics_by_params.csv'), index=False)

    sorted_pd_data = statistics_metrics_df.sort_values(by='final_rank', ascending=True).head(
        top_k if top_k != 'all' else len(statistics_metrics_df))

    best_param_dict_list = sorted_pd_data[list(params_space.keys())].to_dict(orient='records')
    return best_param_dict_list


def find_top_param_dict_list_pareto_by_statistics(pd_data, mode, params_space, top_k, metric_weight_dict,
                                                  metrics_for_pareto_must, stats_weight_dict,
                                                  res_dir):
    logging.info(f"Begin to find top-{top_k} param dict list by statistics in mode={mode} with Pareto optimization...")

    metric_names = list(metric_weight_dict.keys())
    statistics_names = list(stats_weight_dict.keys())
    statistics_metric_names = [f'{metric}_{stat}' for metric in metric_names for stat in statistics_names]
    rank_stat_metric_names = [f'{stat_metric}_rank' for stat_metric in statistics_metric_names]

    # Filter the dataframe to include only mode data and un-pruned params
    valid_params_mode_data = get_valid_params_mode_data(pd_data, mode, params_space)
    # FIXME：计算出不重复的valid_params_dict_list(先用group)
    valid_params_dict_list = valid_params_mode_data[list(params_space.keys())].drop_duplicates() \
        .to_dict(orient='records')
    logging.info(f"len(valid_params_dict_list)={len(valid_params_dict_list)}")

    logging.info(f"Begin to calculate statistics for each parameter combination...")
    statistics_metrics_df = pd.DataFrame(columns=list(params_space.keys()) + statistics_metric_names)
    for param_dict in valid_params_dict_list:
        statistics_dict = calc_statistics(valid_params_mode_data, metric_names, param_dict)
        statistics_metrics_df.loc[len(statistics_metrics_df)] = {**param_dict, **statistics_dict}

    logging.info(f"Begin to calculate ranks for each statistics metric...")
    for stat_metric in statistics_metric_names:
        stat = stat_metric.split('_')[-1]
        metric = '_'.join(stat_metric.split('_')[:-1])
        weight = metric_weight_dict[metric] * stats_weight_dict[stat]
        statistics_metrics_df[f'{stat_metric}_rank'] = statistics_metrics_df[stat_metric].rank(
            ascending=True)  # 所有的都是越小越好
        statistics_metrics_df[f'{stat_metric}_rank'] *= weight

    # 单独计算最全的final_rank并保存
    logging.info(f"Begin to calculate final rank for all statistics metrics...")
    total_weight = sum(metric_weight_dict[metric] * stats_weight_dict[stat]
                       for metric in metric_names for stat in statistics_names)
    statistics_metrics_df['final_rank'] = statistics_metrics_df[rank_stat_metric_names].sum(axis=1) / total_weight
    statistics_metrics_df.to_csv(os.path.join(res_dir, f'_{mode}_statistics_metrics_by_params.csv'), index=False)

    # # Generate all possible metric combinations
    # statistics_metric_names_list = [list(combo) for length in range(1, len(statistics_metric_names) + 1) for combo in
    #                                 itertools.combinations(statistics_metric_names, length)]
    # # 排除不包含must指标的组合
    # rm_index_list = []
    # for idx, stat_metric_names in enumerate(statistics_metric_names_list):
    #     # 要求必须包含的所有must指标 或者 本身是must指标的子集合
    #     if set(stat_metric_names_for_pareto_must).issubset(stat_metric_names):
    #         continue
    #     elif set(stat_metric_names).issubset(stat_metric_names_for_pareto_must):
    #         continue
    #     else:
    #         rm_index_list.append(idx)
    # for idx in rm_index_list[::-1]:
    #     statistics_metric_names_list.pop(idx)

    # Generate all possible metric combinations of metric_names
    metric_names_list = [list(combo) for length in range(1, len(metric_names) + 1) for combo in
                         itertools.combinations(metric_names, length)]
    rm_idxes = []
    for idx, cur_metric_names in enumerate(metric_names_list):
        # 要求必须包含的所有must指标 或者 本身是must指标的子集合
        if set(metrics_for_pareto_must).issubset(cur_metric_names):
            continue
        elif set(cur_metric_names).issubset(metrics_for_pareto_must):
            continue
        else:
            rm_idxes.append(idx)
    for idx in rm_idxes[::-1]:
        metric_names_list.pop(idx)
    # Add statistics for each metric combination
    statistics_metric_names_list = []
    for cur_metric_names in metric_names_list:
        cur_stat_metric_names = []
        for metric in cur_metric_names:
            cur_stat_metric_names.extend([f'{metric}_{stat}' for stat in statistics_names])
        statistics_metric_names_list.append(cur_stat_metric_names)

    logging.info(f"Begin to find Pareto optimal param dict list...")
    pareto_optimal_dict_list = []
    for cur_stat_metric_names in statistics_metric_names_list:
        logging.info(f"cur_stat_metric_names={cur_stat_metric_names}")
        cur_rank_metric_names = [f'{metric}_rank' for metric in cur_stat_metric_names]
        total_weight = sum(metric_weight_dict[metric.split('_')[0]] * stats_weight_dict[metric.split('_')[1]]
                           for metric in cur_stat_metric_names)
        statistics_metrics_df['pareto_rank'] = statistics_metrics_df[cur_rank_metric_names].sum(axis=1) / total_weight
        sorted_pareto_pd_data = statistics_metrics_df.sort_values(by='pareto_rank', ascending=True).head(
            top_k if top_k != 'all' else len(statistics_metrics_df))
        pareto_dict_list = sorted_pareto_pd_data[list(params_space.keys())].to_dict(orient='records')
        pareto_optimal_dict_list.extend(pareto_dict_list)

    logging.info(f"len(pareto_optimal_dict_list)={len(pareto_optimal_dict_list)}")
    return pareto_optimal_dict_list


def find_result_by_param_dict(pd_data, mode, params_space, param_dict):
    metric_name_list = ['mae', 'mse', 'rmse', 'mape', 'mspe']
    grouped_cols = list(params_space.keys()) + metric_name_list
    data_grouped = pd_data[pd_data['mode'] == mode][grouped_cols] \
        .groupby(list(params_space.keys())).mean().reset_index()
    # print('data_grouped', data_grouped)
    # 构造一个列表，包含每个键值对的字典
    filters = [data_grouped[key] == value for key, value in param_dict.items()]
    # 使用all()方法确保所有过滤器都匹配
    matched_rows = data_grouped[np.logical_and.reduce(filters)]
    assert len(matched_rows) == 1, f"len(matched_rows)={len(matched_rows)}"  # 保证只有一个匹配
    row = matched_rows.iloc[0]
    # print('row', row)
    res = {metric_name: row[metric_name] for metric_name in metric_name_list}
    # 加入选出的具体的超参数取值
    res.update({key: row[key] for key in params_space.keys()})
    return res


def calc_improve_percent1(pd_data, mode, params_space, metric_names, val_top1_param_dict):  # FIXME: 顺序！
    # FIXME：之前粒度最小是task，现在是samples
    logging.info(f"Begin to calculate improvement percent1 in mode={mode}...")
    origin_param_dict = get_params_space_and_org()[1]

    result_dict_org = find_result_by_param_dict(pd_data, mode, params_space, origin_param_dict)
    result_dict_our = find_result_by_param_dict(pd_data, mode, params_space, val_top1_param_dict)
    result_dict = {}
    for key in metric_names:
        org, our = result_dict_org[key], result_dict_our[key]
        result_dict[f'org_{key}'] = org
        result_dict[f'our_{key}'] = our
        # 计算improvement_percent （正数越大越好（又因为our越小越好所以是org-our
        # FIXME：
        result_dict[f'improve_percent1_{key}'] = (org - our) / org * 100
        # result_dict[f'improve_percent_{key}'] = (org - our) / ((org + our) / 2) * 100  # SMAPE 平均百分比误差
        # result_dict[f'improve_percent_{key}'] = (org - our) / (org + our) * 100
    return result_dict


# per sample ...
def calc_improve_percent2(pd_data, mode, metric_names, val_top1_param_dict):  # FIXME: 顺序！
    # FIXME：之前粒度最小是task，现在是samples -> 逐sample百分比improve异常值更猛！！！（而且普遍差...）
    logging.info(f"Begin to calculate improvement percent2 in mode={mode}...")
    params_space, origin_param_dict = get_params_space_and_org()

    # Filter data for the specified mode
    data_filtered = pd_data[pd_data['mode'] == mode]
    # Construct masks for the origin and val_top1 param dicts
    mask_origin = np.logical_and.reduce([data_filtered[key] == value for key, value in origin_param_dict.items()])
    mask_val_top1 = np.logical_and.reduce([data_filtered[key] == value for key, value in val_top1_param_dict.items()])
    res = {}
    # Get the metric values for the filtered data
    for metric_name in metric_names:
        logging.info(f"Calculating improvement percent for {metric_name}...")
        origin_metric_values = data_filtered[mask_origin][metric_name].values
        val_top1_metric_values = data_filtered[mask_val_top1][metric_name].values
        assert len(origin_metric_values) == len(val_top1_metric_values), \
            f"Lengths do not match: {len(origin_metric_values)} != {len(val_top1_metric_values)}"
        improve_percent = ((origin_metric_values - val_top1_metric_values) / origin_metric_values * 100).mean()
        res[f"improve_percent2_{metric_name}"] = improve_percent
    return res


def calc_improve_percent_statistics(pd_data, mode, metric_names, val_top1_param_dict):
    logging.info(f"Begin to calculate improvement percent statistics in mode={mode}...")
    params_space, origin_param_dict = get_params_space_and_org()

    # Filter data for the specified mode
    data_filtered = pd_data[pd_data['mode'] == mode]

    # Calculate statistics for origin and val_top1 parameter dictionaries
    origin_statistics = calc_statistics(data_filtered, metric_names, origin_param_dict)
    val_top1_statistics = calc_statistics(data_filtered, metric_names, val_top1_param_dict)

    res = {}
    # Calculate the improvement percent for each statistic
    for metric_name in metric_names:
        logging.info(f"Calculating improvement percent for {metric_name}...")

        origin_mean = origin_statistics[f"{metric_name}_mean"]
        origin_median = origin_statistics[f"{metric_name}_median"]
        origin_std = origin_statistics[f"{metric_name}_std"]
        origin_iqr = origin_statistics[f"{metric_name}_iqr"]
        origin_max = origin_statistics[f"{metric_name}_max"]
        origin_min = origin_statistics[f"{metric_name}_min"]

        val_top1_mean = val_top1_statistics[f"{metric_name}_mean"]
        val_top1_median = val_top1_statistics[f"{metric_name}_median"]
        val_top1_std = val_top1_statistics[f"{metric_name}_std"]
        val_top1_iqr = val_top1_statistics[f"{metric_name}_iqr"]
        val_top1_max = val_top1_statistics[f"{metric_name}_max"]
        val_top1_min = val_top1_statistics[f"{metric_name}_min"]

        improve_percent_mean = (origin_mean - val_top1_mean) / origin_mean * 100
        improve_percent_median = (origin_median - val_top1_median) / origin_median * 100
        improve_percent_std = (origin_std - val_top1_std) / origin_std * 100
        improve_percent_iqr = (origin_iqr - val_top1_iqr) / origin_iqr * 100
        improve_percent_max = (origin_max - val_top1_max) / origin_max * 100
        improve_percent_min = (origin_min - val_top1_min) / origin_min * 100

        res[f"improve_percent_mean_{metric_name}"] = improve_percent_mean
        res[f"improve_percent_median_{metric_name}"] = improve_percent_median
        res[f"improve_percent_std_{metric_name}"] = improve_percent_std
        res[f"improve_percent_iqr_{metric_name}"] = improve_percent_iqr
        res[f"improve_percent_max_{metric_name}"] = improve_percent_max
        res[f"improve_percent_min_{metric_name}"] = improve_percent_min

        # 原始值应该也要做记录到summary。。。以后补上
        # res[f"origin_mean_{metric_name}"] = origin_mean
        # res[f"origin_median_{metric_name}"] = origin_median
        # res[f"origin_std_{metric_name}"] = origin_std
        # res[f"origin_iqr_{metric_name}"] = origin_iqr
        # res[f"origin_max_{metric_name}"] = origin_max
        # res[f"origin_min_{metric_name}"] = origin_min
    return res


def calc_better_draw_percent(pd_data, mode, metric_names, val_top1_param_dict):
    logging.info(f"Begin to calculate better percent in mode={mode}...")

    for metric_name in metric_names:
        assert metric_name in ['mae', 'mse', 'rmse', 'mape', 'mspe'], f"Invalid metric name: {metric_name}"
    params_space, origin_param_dict = get_params_space_and_org()

    # Filter data for the specified mode
    data_filtered = pd_data[pd_data['mode'] == mode]

    # Construct masks for the origin and val_top1 param dicts
    mask_origin = np.logical_and.reduce([data_filtered[key] == value for key, value in origin_param_dict.items()])
    mask_val_top1 = np.logical_and.reduce([data_filtered[key] == value for key, value in val_top1_param_dict.items()])

    res = {}
    # Get the metric values for the filtered data
    for metric_name in metric_names:
        logging.info(f"Calculating better percent for {metric_name}...")
        origin_metric_values = data_filtered[mask_origin][metric_name].values
        val_top1_metric_values = data_filtered[mask_val_top1][metric_name].values

        assert len(origin_metric_values) == len(val_top1_metric_values), \
            f"Lengths do not match: {len(origin_metric_values)} != {len(val_top1_metric_values)}"
        # Calculate the better percent
        better_count = np.sum(val_top1_metric_values < origin_metric_values)
        draw_count = np.sum(val_top1_metric_values == origin_metric_values)  # 好像不可能相等？少数情况会有的
        total_count = len(origin_metric_values)
        better_percent = (better_count / total_count) * 100
        draw_percent = (draw_count / total_count) * 100
        res[f"better_percent_{metric_name}"] = better_percent
        res[f"draw_percent_{metric_name}"] = draw_percent
    return res


#     # 计算val_top1_param_dict和org在test上Bett的sample和Wrse的sample分别提升和降低的比率
#     improve_percent_in_better_and_worse_dict = calc_improve_percent_in_better_and_bad(pd_data, 'test', metric_names,
def calc_improve_percent_in_better_and_worse(pd_data, mode, metric_names, val_top1_param_dict):
    logging.info(f"Begin to calculate improvement percent in better and worse samples in mode={mode}...")
    params_space, origin_param_dict = get_params_space_and_org()

    # Filter data for the specified mode
    data_filtered = pd_data[pd_data['mode'] == mode]
    # Construct masks for the origin and val_top1 param dicts
    mask_origin = np.logical_and.reduce([data_filtered[key] == value for key, value in origin_param_dict.items()])
    mask_val_top1 = np.logical_and.reduce([data_filtered[key] == value for key, value in val_top1_param_dict.items()])
    res = {}
    # Get the metric values for the filtered data
    for metric_name in metric_names:
        logging.info(f"Calculating improvement percent for {metric_name}...")

        origin_metric_values = data_filtered[mask_origin][metric_name].values
        val_top1_metric_values = data_filtered[mask_val_top1][metric_name].values

        assert len(origin_metric_values) == len(val_top1_metric_values), \
            f"Lengths do not match: {len(origin_metric_values)} != {len(val_top1_metric_values)}"

        # Calculate the better and worse samples # mse等越小越好
        better_mask = val_top1_metric_values < origin_metric_values
        worse_mask = val_top1_metric_values > origin_metric_values  # 取等会导致abs(Imp%_MSE(Wrse))偏低

        better_origin_values = origin_metric_values[better_mask]
        better_val_top1_values = val_top1_metric_values[better_mask]

        worse_origin_values = origin_metric_values[worse_mask]
        worse_val_top1_values = val_top1_metric_values[worse_mask]

        # # Calculate the improvement percent
        # if len(better_origin_values) > 0:
        #     improve_percent_better = (
        #             (better_origin_values - better_val_top1_values) / better_origin_values * 100).mean()
        # else:
        #     # improve_percent_better = np.nan  # No better samples
        #     improve_percent_better = 0  # No better samples
        #
        # if len(worse_origin_values) > 0:
        #     improve_percent_worse = ((worse_origin_values - worse_val_top1_values) / worse_origin_values * 100).mean()
        # else:
        #     # improve_percent_bad = np.nan  # No bad samples
        #     improve_percent_worse = 0

        # Calculate the improvement percent
        if len(better_origin_values) > 0:
            mean_better_origin = np.mean(better_origin_values)
            mean_better_val_top1 = np.mean(better_val_top1_values)
            improve_percent_better = (mean_better_origin - mean_better_val_top1) / mean_better_origin * 100
        else:
            improve_percent_better = 0  # No better samples

        if len(worse_origin_values) > 0:
            mean_worse_origin = np.mean(worse_origin_values)
            mean_worse_val_top1 = np.mean(worse_val_top1_values)
            improve_percent_worse = (mean_worse_origin - mean_worse_val_top1) / mean_worse_origin * 100
        else:
            improve_percent_worse = 0  # No worse samples

        res[f"improve_percent_in_better_{metric_name}"] = improve_percent_better
        res[f"improve_percent_in_worse_{metric_name}"] = improve_percent_worse

    return res


def calc_improve_percent_in_hard_medium_easy(pd_data, mode, metric_names, val_top1_param_dict):
    logging.info(f"Begin to calculate improvement percent in hard, medium and easy samples in mode={mode}...")

    params_space, origin_param_dict = get_params_space_and_org()

    # Filter data for the specified mode
    data_filtered = pd_data[pd_data['mode'] == mode]

    # Construct masks for the origin and val_top1 param dicts
    mask_origin = np.logical_and.reduce([data_filtered[key] == value for key, value in origin_param_dict.items()])
    mask_val_top1 = np.logical_and.reduce([data_filtered[key] == value for key, value in val_top1_param_dict.items()])

    res = {}
    # Get the metric values for the filtered data
    for metric_name in metric_names:
        logging.info(f"Calculating improvement percent for {metric_name}...")

        origin_metric_values = data_filtered[mask_origin][metric_name].values
        val_top1_metric_values = data_filtered[mask_val_top1][metric_name].values

        assert len(origin_metric_values) == len(val_top1_metric_values), \
            f"Lengths do not match: {len(origin_metric_values)} != {len(val_top1_metric_values)}"

        # Sort the origin_metric_values to find hard, medium, and easy samples
        sorted_indices = np.argsort(origin_metric_values)
        total_count = len(origin_metric_values)

        # hard_indices = sorted_indices[-(total_count // 3):]
        # medium_indices = sorted_indices[total_count // 3: 2 * total_count // 3]
        # easy_indices = sorted_indices[:total_count // 3]

        hard_indices = sorted_indices[-(total_count // 2):]
        medium_indices = sorted_indices[:]  # 不考虑了
        easy_indices = sorted_indices[:total_count // 2]

        def calculate_improve_percent(indices):
            if len(indices) > 0:
                mean_origin = np.mean(origin_metric_values[indices])
                mean_val_top1 = np.mean(val_top1_metric_values[indices])
                return (mean_origin - mean_val_top1) / mean_origin * 100
            else:
                return 0  # No samples

        improve_percent_easy = calculate_improve_percent(easy_indices)
        improve_percent_medium = calculate_improve_percent(medium_indices)
        improve_percent_hard = calculate_improve_percent(hard_indices)

        res[f"improve_percent_in_easy_{metric_name}"] = improve_percent_easy
        res[f"improve_percent_in_medium_{metric_name}"] = improve_percent_medium
        res[f"improve_percent_in_hard_{metric_name}"] = improve_percent_hard

    return res


def many_plot(pd_data, mode, res_dir):
    # 只需要space而不需要params_list是因为 画图需要遍历所有的组合，而不仅仅是HPO选中的！
    logging.info('\nBegin to plot...')

    params_space, origin_param_dict = get_params_space_and_org()

    # 先把split_idx用mse的mean聚合掉！！！
    grouped_cols = list(params_space.keys()) + ['mae', 'mse', 'rmse', 'mape', 'mspe']
    data_grouped_by_params = pd_data[pd_data['mode'] == mode][grouped_cols] \
        .groupby(list(params_space.keys())).mean().reset_index()
    param_names = list(params_space.keys())

    # 绘制箱线图：
    # 目标：针对每个单独param_name都画一个，展现不同param_value下的mse分布
    # x=str(param_dict), y=mse
    # 注意：期望观察没有聚合后(没有split_idx的情况)，所以使用data_grouped
    # 存在问题，只有个别param_value有柱子！
    # FIXME：速度慢，但是不是很慢，可以接受
    # # 遍历每个参数名
    # logging.info('Creating boxplot...')
    # for param_name in param_names:
    #     param_values = params_space[param_name]['values']
    #     mse_list_list = []
    #     # 遍历每个参数值
    #     for param_value in param_values:
    #         # 根据参数名和参数值过滤数据
    #         # debug ok 注意：使用的是data_grouped_by_params，而不是pd_data！
    #         filtered_data = data_grouped_by_params[data_grouped_by_params[param_name] == param_value]
    #         # 提取过滤后的 MSE 列，并转换为列表
    #         mse_list = filtered_data['mse'].tolist()
    #         mse_list_list.append(mse_list)
    #     # 绘制箱线图
    #     plt.figure(figsize=(8, 6))
    #     plt.boxplot(mse_list_list, labels=param_values, patch_artist=True)
    #     plt.xlabel(f"{param_name}")
    #     plt.ylabel("mse")
    #     plt.title(f"Boxplot of MSE by {param_name}")
    #     plt.savefig(os.path.join(res_dir, mode, f'_boxplot_mse_{param_name}.pdf'), bbox_inches='tight')

    # 绘制分布图：histogram
    # x=param_name y=num_point
    # 目标：针对每个单独param_name都画一个，展现不同param_value下的mse分布
    # 注意：是在同一个split_idx的场景下，找到mse小的排名靠前的param_value，y[sample_r]+1 (限定split_idx之后的排名)
    # 注意：期望观察聚合前(包含split_idx)，所以使用原本的数据pd_data
    # FIXME：速度很慢
    # logging.info('Creating histogram...')
    # top_ratio = 0.25
    # split_idx_list = pd_data[pd_data['mode'] == mode]['split_idx'].unique()
    # for param_name in param_names:
    #     param_values = params_space[param_name]['values']
    #     # 初始化计数数组
    #     num_point_list = np.zeros(len(param_values), dtype=int)
    #     # 预先筛选出每个split_idx的mse
    #     split_mse_dict = {}
    #     for split_idx in split_idx_list:
    #         filter_list = [(pd_data['mode'] == mode), (pd_data['split_idx'] == split_idx)]
    #         mse_list = pd_data[np.logical_and.reduce(filter_list)]['mse'].tolist()
    #         top_k_min_mse = sorted(mse_list)[:ceil(len(mse_list) * top_ratio)]
    #         thresh_mse = top_k_min_mse[-1]
    #         split_mse_dict[split_idx] = thresh_mse
    #     # 遍历每个参数值
    #     for param_value_idx, param_value in enumerate(param_values):
    #         for split_idx in split_idx_list:
    #             thresh_mse = split_mse_dict[split_idx]
    #             filter_list = [(pd_data['mode'] == mode), (pd_data['split_idx'] == split_idx),
    #                            (pd_data[param_name] == param_value)]
    #             filtered_data = pd_data[np.logical_and.reduce(filter_list)]
    #             num_point = len(filtered_data[filtered_data['mse'] <= thresh_mse])
    #             num_point_list[param_value_idx] += num_point
    #     # 绘制分布图
    #     plt.figure(figsize=(8, 6))
    #     names = [f"{param_value}" for param_value in param_values]
    #     plt.bar(names, num_point_list, width=0.1)
    #     plt.xlabel(f"{param_name}")
    #     plt.ylabel("num_point")
    #     plt.title(f"Num of Points by {param_name}")
    #     plt.savefig(os.path.join(res_dir, mode, f'_num_point_{param_name}.pdf'), bbox_inches='tight')
    #     plt.close()

    # 平行坐标图！
    # n*x=n*param_name+mse
    # 目标：展示不同超参数组合下的 平均mse （使用聚合后的数据）
    # 创建维度对象
    logging.info('Creating parallel coordinates plot...')
    dimensions = []
    logging.info('Adding dimensions for hyperparameters...')
    for param_name in param_names:
        param_values = params_space[param_name]['values']
        param_type = params_space[param_name]['type']
        if param_type in ['float', 'int']:
            range_min = min(param_values)
            range_max = max(param_values)
            values = data_grouped_by_params[param_name].tolist()
            dimension = dict(
                range=[range_min, range_max],
                label=param_name,
                values=values,
            )
        elif param_type == 'str':
            range_min = 0
            range_max = len(param_values) - 1  # Use index as range for string values
            values = [param_values.index(value) for value in data_grouped_by_params[param_name].tolist()]
            texts = param_values
            dimension = dict(
                range=[range_min, range_max],
                label=param_name,
                values=values,
                tickvals=list(range(len(texts))),  # Use index as tickvals
                ticktext=texts,
            )
        else:
            raise ValueError(f"Unknown type: {param_type}")
        dimensions.append(dimension)
    # 加上mse等metrics的dimension
    logging.info('Adding dimensions for metrics...')
    for metric in ['mae', 'mse', 'rmse', 'mape', 'mspe']:
        dimensions.append(dict(
            range=[min(data_grouped_by_params[metric]), max(data_grouped_by_params[metric])],
            label=metric,
            values=data_grouped_by_params[metric].tolist()
        ))
    # 创建数据对象
    logging.info('Creating data object...')
    data = [
        go.Parcoords(
            line=dict(color='blue'),
            dimensions=dimensions,
            labelfont=dict(size=8, color='black'),
            tickfont=dict(size=8, color='black'),
        )
    ]
    # 创建布局对象
    logging.info('Creating layout object...')
    layout = go.Layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        title=f"Parallel Coordinates Plot of MSE by Hyperparameters",
    )
    # 创建图对象
    logging.info('Creating figure object...')
    fig = go.Figure(data=data, layout=layout)
    # 保存图对象
    # py.offline.plot(fig, filename=os.path.join(res_dir, mode, '_parallel_coordinates_plot.html'))
    logging.info('Writing HTML...')
    fig.write_html(os.path.join(res_dir, mode, '_parallel_coordinates_plot.html'), auto_open=False)

    plot_hpo_progress(pd_data, res_dir, mode, params_space)

    logging.info('Plotting finished!')


def plot_hpo_progress(pd_data, res_dir, mode, params_space):  # 仅画mean_mse
    # FIXME:提前被prune掉的也被画上了。。。

    logging.info('Creating HPO progress plot...')
    # 获取未被剪枝的params的个数：
    total_num_combinations = len(pd_data[pd_data['mode'] == mode].groupby(list(params_space.keys()))
                                 .agg({'mse': 'mean'})['mse'].reset_index())
    logging.info(f'total_num_combinations={total_num_combinations}')

    # 去除了被剪枝的params数据
    valid_params_mode_data = get_valid_params_mode_data(pd_data, mode, params_space)
    data_grouped = valid_params_mode_data.groupby(list(params_space.keys())).agg({'mse': 'mean'}).reset_index()
    valid_num_combinations = len(data_grouped)
    valid_mse_values = data_grouped['mse'].values
    logging.info(f'valid_num_combinations={valid_num_combinations}')
    logging.info(f'valid mse_values={valid_mse_values}')

    # 去除掉异常值以免影响画图：-> 画图时限制即可
    lower_bound = 0  # 无限制
    upper_bound = min(valid_mse_values) * 30  # 一个数量级多一点
    # mse_values = np.clip(mse_values, lower_bound, upper_bound)
    # logging.info(f'mse_values after clipping={mse_values}')

    # 初始化最佳 MSE 的跟踪列表
    best_mse_values = []
    best_mse = float('inf')
    for mse in valid_mse_values:
        if mse < best_mse:
            best_mse = mse
        best_mse_values.append(best_mse)
    logging.info(f'best_mse_values={best_mse_values}')

    plt.figure(figsize=(10, 6))

    # 绘制散点图
    plt.scatter(range(1, valid_num_combinations + 1), valid_mse_values, label='MSE', color='blue', s=10)

    # 绘制最佳 MSE 折线图
    plt.plot(range(1, valid_num_combinations + 1), best_mse_values, label='Best MSE', color='red')

    plt.xlabel('Number of parameter combinations')
    plt.ylabel('MSE (log scale)')
    plt.yscale('log')  # 设置 y 轴为对数刻度
    plt.ylim(lower_bound, upper_bound)  # 限制 y 轴范围
    plt.title(f'HPO Progress: MSE over parameter combinations {mode} Mode (Total: {total_num_combinations})')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plot_path = os.path.join(res_dir, mode, '_hpo_progress_plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    logging.info(f'HPO progress plot saved to {plot_path}')


# # 画直方图直观比较org和our的metric的分布差异
# plot_metric_hist(pd_data, 'test', params_space, 'mse', res_dir,
#                  val_top1_param_dict=val_top1_param_dict, origin_param_dict=origin_param_dict)
def plot_metric_hist_comparison(pd_data, mode, metric_names, res_dir, val_top1_param_dict):
    logging.info(f"Begin to plot histograms for {metric_names} in mode={mode}...")

    params_space, origin_param_dict = get_params_space_and_org()

    mode_data = pd_data[pd_data['mode'] == mode]
    mask_origin = np.logical_and.reduce([mode_data[key] == value for key, value in origin_param_dict.items()])
    mask_val_top1 = np.logical_and.reduce([mode_data[key] == value for key, value in val_top1_param_dict.items()])

    for metric_name in metric_names:
        logging.info(f"Plotting histogram for {metric_name}...")
        origin_metric_values = mode_data[mask_origin][metric_name].values
        val_top1_metric_values = mode_data[mask_val_top1][metric_name].values

        assert len(origin_metric_values) == len(val_top1_metric_values), \
            f"Lengths do not match: {len(origin_metric_values)} != {len(val_top1_metric_values)}"

        plt.figure(figsize=(12, 6))

        # # Plot histograms for origin and val_top1 metric values
        # plt.hist(origin_metric_values, bins=50, alpha=0.5, label='Origin', color='blue', log=True)
        # plt.hist(val_top1_metric_values, bins=50, alpha=0.5, label='Val Top1', color='orange', log=True)

        # Compute histograms
        bins = np.linspace(min(origin_metric_values.min(), val_top1_metric_values.min()),
                           max(origin_metric_values.max(), val_top1_metric_values.max()), 50)
        origin_hist, bins = np.histogram(origin_metric_values, bins=bins)
        val_top1_hist, _ = np.histogram(val_top1_metric_values, bins=bins)
        # Plot histograms with same bar width
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = (bins[1] - bins[0]) * 1  # Set bar width to 40% of bin width # 中间白色太亮眼 0.4 0.5 1
        plt.bar(bin_centers - width / 2 * 0, origin_hist, width=width, label='Origin', color='orange', alpha=0.5)
        plt.bar(bin_centers + width / 2 * 0, val_top1_hist, width=width, label='Val Top1', color='blue', alpha=0.5)
        # FIXME: log or not
        # plt.xscale('log') # bin宽度形状会变得不一致
        # plt.yscale('log') # 面积比例会不一致？
        # 看不出来明显优势？？？ -》 xlog才能显示出大小值的差异？
        # plt.bar(bin_centers - width / 2, origin_hist, width=width, alpha=0.5, label='Origin', color='blue')
        # plt.bar(bin_centers + width / 2, val_top1_hist, width=width, alpha=0.5, label='Val Top1', color='orange')

        plt.xlabel(metric_name)
        plt.ylabel('Frequency')
        plt.title(f'{metric_name} Distribution in {mode} Mode')
        plt.legend(loc='upper right')

        plot_path = os.path.join(res_dir, f'_{mode}_{metric_name}_histogram.png')
        plt.savefig(plot_path)
        logging.info(f"Histogram saved to {plot_path}")
        plt.close()


# atexit handler
def atexit_handler(res_dir):
    # Function to write the return code to a file
    def write_return_code(return_code, status_file):
        with open(status_file, 'w') as f:
            f.write(str(return_code))

    # Check if an exception occurred
    exc_type, exc_value, exc_traceback = sys.exc_info()
    status_file = os.path.join(res_dir, 'return_code.txt')
    if exc_type is not None:
        # Exception occurred, write a non-zero return code
        write_return_code(1, status_file)
    else:
        # No exception, write a zero return code
        write_return_code(0, status_file)
