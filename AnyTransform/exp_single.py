import argparse
import atexit
import cProfile
import logging
import os
import pstats
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from AnyTransform.parser import *
from AnyTransform.config import *
from AnyTransform import metric
from AnyTransform.augmentor import Augmentor
from AnyTransform.pipeline import *
from AnyTransform.terminator import *
from dataset import *
from model import get_model
from transforms import Sampler, Warper, Trimmer, Decomposer, Aligner, Normalizer, Inputer, Differentiator
from tuners import MyBatchTuner, OptunaTuner
from utils import time_start, log_time_delta

import itertools

import os
import sys
from math import ceil


# trimmer, sampler, inputer, denoiser, warper, decomposer, differentiator, normalizer, aligner, model
# def infer3(history_seqs, model, dataset, target_column, patch_len, pred_len, mode, sampler_factor, trimmer_seq_len,
#            aligner_mode, aligner_method, normalizer_method, normalizer_mode, normalizer_ratio, inputer_detect_method,
#            inputer_fill_method, warper_method, decomposer_period, decomposer_components, differentiator_n,
#            denoiser_method, clip_factor):
def ablation_substitute(kwargs):
    params_space, origin_param_dict = get_params_space_and_org()
    if ablation == 'Trimmer' or ablation == 'Context':
        kwargs['trimmer_seq_len'] = origin_param_dict['trimmer_seq_len']
    elif ablation == 'Sampler' or ablation == 'Context':
        kwargs['sampler_factor'] = origin_param_dict['sampler_factor']
    elif ablation == 'Aligner' or ablation == 'Context':
        kwargs['aligner_mode'] = origin_param_dict['aligner_mode']
        kwargs['aligner_method'] = origin_param_dict['aligner_method']

    elif ablation == 'Normalizer' or ablation == 'Range':
        kwargs['normalizer_method'] = origin_param_dict['normalizer_method']
        kwargs['normalizer_mode'] = origin_param_dict['normalizer_mode']
        kwargs['normalizer_ratio'] = origin_param_dict['normalizer_ratio']
    elif ablation == 'Warper' or ablation == 'Range':
        kwargs['warper_method'] = origin_param_dict['warper_method']
    elif ablation == 'Differentiator' or ablation == 'Range':
        kwargs['differentiator_n'] = origin_param_dict['differentiator_n']

    elif ablation == 'Inputer' or ablation == 'Anomaly':
        kwargs['inputer_detect_method'] = origin_param_dict['inputer_detect_method']
        kwargs['inputer_fill_method'] = origin_param_dict['inputer_fill_method']
    elif ablation == 'Denoiser' or ablation == 'Anomaly':
        kwargs['denoiser_method'] = origin_param_dict['denoiser_method']
    elif ablation == 'Clipper' or ablation == 'Anomaly':
        kwargs['clip_factor'] = origin_param_dict['clip_factor']

    elif ablation == 'Pipeline':
        kwargs['pipeline_name'] = origin_param_dict['pipeline_name']
    return kwargs


def train_or_test_or_val(
        patch_len, pred_len, max_seq_len,  # 相对固定
        model, dataset, target_column, mode, num_sample,  # 相对固定
        augmentor, batch_sizes,  # 新增的参数!!!!!!!!!
        params_space, tuner, pruner_report_mode, pruner_metric_mode, num_params, terminator_manager: TerminatorManager,
        pd_data, res_dir,  # 可变
        tr: TimeRecorder,
        args=None
):
    logging.info("###############################################")
    logging.info(f"Begin to {mode}...")

    mode_for_data = 'train' if mode != 'test' else 'test'  # !!!!!!!! val模式使用的是train的data！！！！
    custom_dataset = CustomDataset(dataset, mode_for_data, target_column, max_seq_len, pred_len, augmentor, num_sample)
    # dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)  # FIXME: 按道理test的batch_size=1但是太慢
    batch_sampler = DynamicBatchSampler(custom_dataset, batch_sizes)  # 动态的batch_size
    dataloader = DataLoader(custom_dataset, batch_sampler=batch_sampler)

    standard_scaler = dataset.get_mode_scaler(mode_for_data, 'standard', target_column)
    original_aug_method = list(augmentor.aug_method_dict.keys())[augmentor.aug_idx]

    bar1 = tqdm(range(num_params), desc='Processing Params', ncols=100) \
        if mode != 'test' else range(num_params)
    for _ in bar1:
        print()
        param_dict = tuner.ask()
        logging.info(f"param_dict={param_dict}")

        max_step_idx = len(dataloader) - 1
        augmentor.reset_aug_method(original_aug_method)  # 确保不同param_dict下的同一数据的augment方法一致
        if mode == 'test':
            assert original_aug_method == 'none', f"original_aug_method={original_aug_method}"
            assert augmentor.mode == 'fix', f"augmentor.mode={augmentor.mode}"

        should_prune = False
        batch_mse_list = []  # !!!!!! 存储当前param_dict的mse列表，无论是否被prune都计算的全split_idx的mse
        bar2 = tqdm(enumerate(dataloader), desc='Processing Split Idxes', ncols=100, total=len(dataloader)) \
            if mode == 'test' else enumerate(dataloader)
        for step_idx, (split_idxes, aug_methods, historys, labels) in bar2:
            assert should_prune is False, "should_prune should be False at the beginning of each split_idx loop"
            print() if mode == 'test' else None
            logging.info(f"{split_idxes.shape=}, {historys.shape=}, {labels.shape=}")
            logging.info(f"step ratio: {step_idx + 1}/{max_step_idx + 1}")

            # aug产生的四维数据 batch_size,aug,time,feature -> 合并前两维并限制 shape[0] 不超过800
            assert len(historys.shape) == 4 and len(labels.shape) == 4, \
                f"historys.shape={historys.shape}, labels.shape={labels.shape}"
            if mode == 'test':  # test 可不能有非none的aug
                assert historys.shape[1] == 1, f"historys.shape[1]={historys.shape[1]}"  # no aug for test

            # transpose: batch,aug,time,feature -> aug,batch,time,feature (靠前的aug更重要->none (真不是！！！均衡更重要！！！
            # historys = historys.permute(1, 0, 2, 3)
            # labels = labels.permute(1, 0, 2, 3)

            historys = historys.reshape(-1, historys.shape[2], historys.shape[3])
            labels = labels.reshape(-1, labels.shape[2], labels.shape[3])

            # max_batch_size
            max_batch_size_for_mode = 1000 if mode == 'train' else np.inf  # FIXME: 1000 for train fast for Uni2ts
            max_batch_size_for_cuda = get_max_batch_size_for_cuda(model.model_name)
            max_batch_size = min(max_batch_size_for_mode, max_batch_size_for_cuda)
            logging.info(f"max_batch_size_for_cuda={max_batch_size_for_cuda}")
            if mode == 'test':  # test 可不能被截断
                if args.model_name == 'Arima':
                    max_batch_size_for_cuda = historys.shape[0]
                assert historys.shape[0] <= max_batch_size_for_cuda, f"historys.shape[0]={historys.shape[0]}"
            # train/val 只选择最多max_batch_size个样本（均匀选取，保证每个params给到idx的相同
            if historys.shape[0] > max_batch_size:  # FIXME: 配合prune的step
                # logging.info(f"historys.shape[0]={historys.shape[0]}")
                # selected_idxes = np.linspace(0, historys.shape[0] - 1, max_batch_size, dtype=int)
                # historys = historys[selected_idxes]
                # labels = labels[selected_idxes]
                historys, labels = historys[:max_batch_size], labels[:max_batch_size]
                split_idxes, aug_methods = split_idxes[:max_batch_size], aug_methods[:max_batch_size]
            actual_batch_size = historys.shape[0]
            logging.debug(f"aug_methods={aug_methods}")

            # 转化成numpy
            historys, labels = historys.numpy(), labels.numpy()

            # 用 adaptive_infer 推理
            t_infer = time_start()
            kwargs = param_dict.copy()
            kwargs = ablation_substitute(kwargs)
            kwargs.update({'history_seqs': historys, 'model': model, 'dataset': dataset,
                           'target_column': target_column, 'patch_len': patch_len, 'pred_len': pred_len, 'mode': mode})
            preds, process_dur, model_dur = adaptive_infer(args,**kwargs)
            log_time_delta(t_infer, "infer")

            # 如果有nan或inf则clip
            if np.isnan(preds).any() or np.isinf(preds).any():
                my_clip(historys, preds, nan_inf_clip_factor=nan_inf_clip_factor)

            t_metric = time_start()
            # 标准化处理
            _preds = standard_scaler.transform(preds.reshape(-1, 1)).reshape(preds.shape)
            _labels = standard_scaler.transform(labels.reshape(-1, 1)).reshape(labels.shape)

            # 计算该batch全量split_idx的指标->多batch的mean给tuner做final
            mae, mse, rmse, mape, mspe = metric(_preds, _labels)
            batch_mse_list.append(mse)

            # 计算指标
            # tr.time_end() if tr is not None and mode == 'test' else None  # test不需要计算这部分 每个split的结果还是要记的
            if pruner_report_mode == 'batch':
                # mae, mse, rmse, mape, mspe = metric(_preds, _labels) # 沿用上面的
                index_str = str(len(pd_data)).zfill(5)
                result_dict = {**param_dict, 'split_idx': str(split_idxes.tolist()), 'idx': index_str, 'mode': mode,
                               'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe, 'aug_method': 'mix'}
                pd_data.loc[len(pd_data)] = result_dict
                logging.info(f"cur_mse={mse}")

                if mode != 'test':  # !!!
                    _mse = mse if pruner_metric_mode == 'cur' \
                        else find_result_by_param_dict(pd_data, mode, params_space, param_dict)['mse']
                    tuner.report(param_dict, _mse, step_idx)
                    logging.info(f"{pruner_metric_mode}_mse={_mse}")
                    if step_idx != max_step_idx and tuner.should_prune(param_dict):
                        logging.info(f"Prune at step_idx={step_idx} with {pruner_metric_mode}_mse={_mse}")
                        should_prune = True
            elif pruner_report_mode == 'single':
                assert len(batch_sizes) == 1, "single report mode only support single batch size"
                # batch_results = [None] * len(split_idxes)  # 也不一定就比原生写pd快 FIXME:
                # FIXME: actual_batch_size才是真实的batch_size，不然会出现空的nan的数据条目
                # ps:（如果还有考虑all的aug方法重复split_idx就太麻烦了。。。（重复好像也没事，一条新记录
                batch_results = [None] * actual_batch_size
                for i in range(actual_batch_size):
                    split_idx = split_idxes[i]
                    # FIXME: 决定被prune但已经计算出后面split_idx的metric要不要记录影响后续prune难度(影响选top
                    # 。。。 break会少了记录？，但是不break跟原来batch的选top逻辑就一样了
                    if should_prune:
                        break
                    mae, mse, rmse, mape, mspe = metric(_preds[i:i + 1], _labels[i:i + 1])
                    aug_method = aug_methods[i]
                    index_str = str(len(pd_data) + i).zfill(5)
                    result_dict = {**param_dict, 'split_idx': str([int(split_idx)]), 'idx': index_str, 'mode': mode,
                                   'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe,
                                   'aug_method': aug_method}
                    batch_results[i] = result_dict
                    if mode != 'test':
                        _mse = mse  # 不支持mean，分离读写场景下有点复杂
                        tuner.report(param_dict, _mse, i)
                        # logging.debug(f"{pruner_metric_mode}_mse={_mse}")
                        if tuner.should_prune(param_dict):
                            logging.info(f"Prune at step_idx={step_idx} split_idx={split_idx} "
                                         f"with {pruner_metric_mode}_mse={_mse}")
                            should_prune = True
                valid_batch_results = [result for result in batch_results if result]
                # logging.debug(f"valid_batch_results={valid_batch_results}")
                # cat会导致外部pd_data跟内部不一致！
                pd_data = pd.concat([pd_data, pd.DataFrame(valid_batch_results)], ignore_index=True)
                # # # 使用 loc 方法批量插入数据
                # # pd_data.loc[len(pd_data)] = result_dict
                # pd_data.loc[len(pd_data):len(pd_data) + len(valid_batch_results)] = pd.DataFrame(valid_batch_results)
            else:
                raise ValueError(f"Unknown pruner_report_mode: {pruner_report_mode}")
            log_time_delta(t_metric, "metric")

            # if step == max_step and mode != 'train':
            cur_batch_size = len(split_idxes)

            selected_split_idx_num = 0
            if mode == 'val':
                selected_split_idx_num = 1
            if mode == 'test':  # 每个batch有多次对比
                selected_split_idx_num = 1  # 参考data画图和以前的pdf数据吧
            # 平均选取
            selected_split_idx_num = min(selected_split_idx_num, cur_batch_size)
            selected_batch_idxes = np.linspace(0, cur_batch_size - 1, selected_split_idx_num, dtype=int) \
                if selected_split_idx_num != 0 else []
            # [45210] [45354] weather
            # FIXME: 补充画图的split_idx
            if mode == 'test':
                if data_name == 'Weather':
                    selected_batch_idxes = list(selected_batch_idxes)
                    if 45210 in split_idxes:
                        selected_batch_idxes.append(np.where(split_idxes == 45210)[0][0])
                    if 45354 in split_idxes:
                        selected_batch_idxes.append(np.where(split_idxes == 45354)[0][0])
                    selected_batch_idxes = np.array(selected_batch_idxes)
                if data_name == 'Traffic':  # [15832]
                    selected_batch_idxes = list(selected_batch_idxes)
                    if 15832 in split_idxes:
                        selected_batch_idxes.append(np.where(split_idxes == 15832)[0][0])
                    selected_batch_idxes = np.array(selected_batch_idxes)
            tr.time_end() if tr is not None else None
            for i in selected_batch_idxes:
                seq = historys[i, -3 * pred_len:, 0]
                pred = preds[i, :, 0]
                label = labels[i, :, 0]
                pred_line = np.concatenate((seq, pred), axis=0)
                label_line = np.concatenate((seq, label), axis=0)
                plt.figure(figsize=(ceil(len(label_line) / patch_len) * 5, 5))
                plt.plot(pred_line, label="pred", linestyle='--', color='blue')
                plt.plot(label_line, label="label", linestyle='-', color='orange')
                plt.legend()
                # 同一个样本的不同处理方式 方便比较
                plt.savefig(os.path.join(res_dir, mode, f'{split_idxes.tolist()[i]}-{index_str}.pdf'),
                            bbox_inches='tight')
            tr.time_start() if tr is not None else None
            if should_prune:
                break

        # FIXME：无论是否被prune，都要告诉tuner这个param_dict已经结束了，有点问题。。。mse； Ok:在筛选top1时会考虑max_step
        # FIXME: 这里的mean_mse在prune后不是在所有的split_idx上计算的！！！
        # FIXME：更高保真的mean_mse计算方式 -> 所有过去历史batch的全量split_idx的mse的mean
        # 不真实的mse->对TPE影响较大（单batch用batch_mse_list则不受坏影响
        if should_prune:
            final_mse = np.mean(batch_mse_list)
        else:
            final_mse = find_result_by_param_dict(pd_data, mode, params_space, param_dict)['mse']
        tuner.tell(param_dict, final_mse)
        logging.info(f"final_mse={final_mse}")
        terminate_flag = terminator_manager.update_and_check(final_mse, should_prune) \
            if terminator_manager is not None else False

        # # FIXME: 尝试用better_percent替代！！！！！！！！！intermediate怎么办... cur不好弄
        # better_percent_mse = calc_better_draw_percent(pd_data, mode, ['mse'], get_params_space_and_org()[1],
        #                                               param_dict)['better_percent_mse']
        # logging.info(f"better_percent_mse={better_percent_mse}")
        # tuner.tell(param_dict, better_percent_mse)
        # terminate_flag = terminator_manager.update_and_check(0, should_prune) \
        #     if terminator_manager is not None else False
        # terminate_flag = False

        if terminate_flag:
            logging.info(f"Experiment terminated by terminator!")
            break
    tr.time_end() if tr is not None else None
    t = time_start()
    pd_data.to_csv(os.path.join(res_dir, 'pd_data.csv'))
    grouped_cols = list(params_space.keys()) + ['mae', 'mse', 'rmse', 'mape', 'mspe'] + ['split_idx', 'idx']
    agg_dict = {'mae': 'mean', 'mse': 'mean', 'rmse': 'mean', 'mape': 'mean', 'mspe': 'mean',
                'split_idx': 'last', 'idx': 'last'}
    data_grouped_by_params = pd_data[pd_data['mode'] == mode][grouped_cols] \
        .groupby(list(params_space.keys())).agg(agg_dict).reset_index()
    # 没有区分prune，但是后续是pd_data再算top...
    data_grouped_by_params.to_csv(os.path.join(res_dir, f'_{mode}_data_grouped_by_params.csv'))
    log_time_delta(t, f"save {mode} data_grouped_by_params")
    tr.time_start() if tr is not None else None
    return pd_data


def train_or_test_or_val_cov(
        patch_len, pred_len, max_seq_len,  # 相对固定
        model, dataset, target_column, mode, num_sample,  # 相对固定
        augmentor, batch_sizes,  # 新增的参数!!!!!!!!!
        params_space, tuner, pruner_report_mode, pruner_metric_mode, num_params, terminator_manager: TerminatorManager,
        pd_data, res_dir,  # 可变
        tr: TimeRecorder
):
    logging.info("###############################################")
    logging.info(f"Begin to {mode}...")
    
    mode_for_data = 'train' if mode != 'test' else 'test'  # !!!!!!!! val模式使用的是train的data！！！！
    
    custom_dataset = CustomDatasetCov(dataset, mode_for_data, target_column, max_seq_len, pred_len, augmentor,
                                      num_sample)
    
    # dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)  # FIXME: 按道理test的batch_size=1但是太慢
    batch_sampler = DynamicBatchSampler(custom_dataset, batch_sizes)  # 动态的batch_size
    dataloader = DataLoader(custom_dataset, batch_sampler=batch_sampler)
    
    standard_scaler = dataset.get_mode_scaler(mode_for_data, 'standard', target_column)
    original_aug_method = list(augmentor.aug_method_dict.keys())[augmentor.aug_idx]
    
    bar1 = tqdm(range(num_params), desc='Processing Params', ncols=100) \
        if mode != 'test' else range(num_params)
    for _ in bar1:
        print()
        param_dict = tuner.ask()
        logging.info(f"param_dict={param_dict}")
        
        max_step_idx = len(dataloader) - 1
        augmentor.reset_aug_method(original_aug_method)  # 确保不同param_dict下的同一数据的augment方法一致
        if mode == 'test':
            assert original_aug_method == 'none', f"original_aug_method={original_aug_method}"
            assert augmentor.mode == 'fix', f"augmentor.mode={augmentor.mode}"
        
        should_prune = False
        batch_mse_list = []  # !!!!!! 存储当前param_dict的mse列表，无论是否被prune都计算的全split_idx的mse
        bar2 = tqdm(enumerate(dataloader), desc='Processing Split Idxes', ncols=100, total=len(dataloader)) \
            if mode == 'test' else enumerate(dataloader)
        for step_idx, (split_idxes, aug_methods, historys, labels, cov_historys, cov_labels) in bar2:
            assert should_prune is False, "should_prune should be False at the beginning of each split_idx loop"
            print() if mode == 'test' else None
            logging.info(f"{split_idxes.shape=}, {historys.shape=}, {labels.shape=}, {cov_historys.shape=}, {cov_labels.shape=}")
            logging.info(f"step ratio: {step_idx + 1}/{max_step_idx + 1}")
            
            # aug产生的四维数据 batch_size,aug,time,feature -> 合并前两维并限制 shape[0] 不超过800
            assert len(historys.shape) == 4 and len(labels.shape) == 4, \
                f"historys.shape={historys.shape}, labels.shape={labels.shape}"
            if mode == 'test':  # test 可不能有非none的aug
                assert historys.shape[1] == 1, f"historys.shape[1]={historys.shape[1]}"  # no aug for test
            
            # transpose: batch,aug,time,feature -> aug,batch,time,feature (靠前的aug更重要->none (真不是！！！均衡更重要！！！
            # historys = historys.permute(1, 0, 2, 3)
            # labels = labels.permute(1, 0, 2, 3)
            
            historys = historys.reshape(-1, historys.shape[2], historys.shape[3])
            labels = labels.reshape(-1, labels.shape[2], labels.shape[3])
            cov_historys = cov_historys.reshape(-1, cov_historys.shape[2], cov_historys.shape[3])
            cov_labels = cov_labels.reshape(-1, cov_labels.shape[2], cov_labels.shape[3])
            
            # max_batch_size
            max_batch_size_for_mode = 1000 if mode == 'train' else np.inf  # FIXME: 1000 for train fast for Uni2ts
            max_batch_size_for_cuda = get_max_batch_size_for_cuda(model.model_name)
            max_batch_size = min(max_batch_size_for_mode, max_batch_size_for_cuda)
            logging.info(f"max_batch_size_for_cuda={max_batch_size_for_cuda}")
            if mode == 'test':  # test 可不能被截断
                assert historys.shape[0] <= max_batch_size_for_cuda, f"historys.shape[0]={historys.shape[0]}"
            # train/val 只选择最多max_batch_size个样本（均匀选取，保证每个params给到idx的相同
            if historys.shape[0] > max_batch_size:  # FIXME: 配合prune的step
                # logging.info(f"historys.shape[0]={historys.shape[0]}")
                # selected_idxes = np.linspace(0, historys.shape[0] - 1, max_batch_size, dtype=int)
                # historys = historys[selected_idxes]
                # labels = labels[selected_idxes]
                historys, labels = historys[:max_batch_size], labels[:max_batch_size]
                cov_historys, cov_labels = cov_historys[:max_batch_size], cov_labels[:max_batch_size]
                split_idxes, aug_methods = split_idxes[:max_batch_size], aug_methods[:max_batch_size]
            actual_batch_size = historys.shape[0]
            logging.debug(f"aug_methods={aug_methods}")
            
            # 转化成numpy
            historys, labels = historys.numpy(), labels.numpy()
            cov_historys, cov_labels = cov_historys.numpy(), cov_labels.numpy()
            
            # 用 adaptive_infer 推理
            t_infer = time_start()
            kwargs = param_dict.copy()
            kwargs = ablation_substitute(kwargs)
            kwargs.update({'history_seqs': historys, 'cov_history_seqs': cov_historys, 'model': model, 'dataset': dataset,
                           'target_column': target_column, 'patch_len': patch_len, 'pred_len': pred_len, 'mode': mode,
                           'pipeline_name': 'infer_cov'})
            preds, process_dur, model_dur = adaptive_infer(args,**kwargs)
            log_time_delta(t_infer, "infer")
            
            # 如果有nan或inf则clip
            if np.isnan(preds).any() or np.isinf(preds).any():
                my_clip(historys, preds, nan_inf_clip_factor=nan_inf_clip_factor)
            
            t_metric = time_start()
            # 标准化处理
            _preds = standard_scaler.transform(preds.reshape(-1, 1)).reshape(preds.shape)
            _labels = standard_scaler.transform(labels.reshape(-1, 1)).reshape(labels.shape)
            
            # 计算该batch全量split_idx的指标->多batch的mean给tuner做final
            mae, mse, rmse, mape, mspe = metric(_preds, _labels)
            batch_mse_list.append(mse)
            
            # 计算指标
            # tr.time_end() if tr is not None and mode == 'test' else None  # test不需要计算这部分 每个split的结果还是要记的
            if pruner_report_mode == 'batch':
                # mae, mse, rmse, mape, mspe = metric(_preds, _labels) # 沿用上面的
                index_str = str(len(pd_data)).zfill(5)
                result_dict = {**param_dict, 'split_idx': str(split_idxes.tolist()), 'idx': index_str, 'mode': mode,
                               'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe, 'aug_method': 'mix'}
                pd_data.loc[len(pd_data)] = result_dict
                logging.info(f"cur_mse={mse}")
                
                if mode != 'test':  # !!!
                    _mse = mse if pruner_metric_mode == 'cur' \
                        else find_result_by_param_dict(pd_data, mode, params_space, param_dict)['mse']
                    tuner.report(param_dict, _mse, step_idx)
                    logging.info(f"{pruner_metric_mode}_mse={_mse}")
                    if step_idx != max_step_idx and tuner.should_prune(param_dict):
                        logging.info(f"Prune at step_idx={step_idx} with {pruner_metric_mode}_mse={_mse}")
                        should_prune = True
            elif pruner_report_mode == 'single':
                assert len(batch_sizes) == 1, "single report mode only support single batch size"
                # batch_results = [None] * len(split_idxes)  # 也不一定就比原生写pd快 FIXME:
                # FIXME: actual_batch_size才是真实的batch_size，不然会出现空的nan的数据条目
                # ps:（如果还有考虑all的aug方法重复split_idx就太麻烦了。。。（重复好像也没事，一条新记录
                batch_results = [None] * actual_batch_size
                for i in range(actual_batch_size):
                    split_idx = split_idxes[i]
                    # FIXME: 决定被prune但已经计算出后面split_idx的metric要不要记录影响后续prune难度(影响选top
                    # 。。。 break会少了记录？，但是不break跟原来batch的选top逻辑就一样了
                    if should_prune:
                        break
                    mae, mse, rmse, mape, mspe = metric(_preds[i:i + 1], _labels[i:i + 1])
                    aug_method = aug_methods[i]
                    index_str = str(len(pd_data) + i).zfill(5)
                    result_dict = {**param_dict, 'split_idx': str([int(split_idx)]), 'idx': index_str, 'mode': mode,
                                   'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe,
                                   'aug_method': aug_method}
                    batch_results[i] = result_dict
                    if mode != 'test':
                        _mse = mse  # 不支持mean，分离读写场景下有点复杂
                        tuner.report(param_dict, _mse, i)
                        # logging.debug(f"{pruner_metric_mode}_mse={_mse}")
                        if tuner.should_prune(param_dict):
                            logging.info(f"Prune at step_idx={step_idx} split_idx={split_idx} "
                                         f"with {pruner_metric_mode}_mse={_mse}")
                            should_prune = True
                valid_batch_results = [result for result in batch_results if result]
                # logging.debug(f"valid_batch_results={valid_batch_results}")
                # cat会导致外部pd_data跟内部不一致！
                pd_data = pd.concat([pd_data, pd.DataFrame(valid_batch_results)], ignore_index=True)
                # # # 使用 loc 方法批量插入数据
                # # pd_data.loc[len(pd_data)] = result_dict
                # pd_data.loc[len(pd_data):len(pd_data) + len(valid_batch_results)] = pd.DataFrame(valid_batch_results)
            else:
                raise ValueError(f"Unknown pruner_report_mode: {pruner_report_mode}")
            log_time_delta(t_metric, "metric")
            
            # if step == max_step and mode != 'train':
            cur_batch_size = len(split_idxes)
            
            selected_split_idx_num = 0
            if mode == 'val':
                selected_split_idx_num = 1
            if mode == 'test':  # 每个batch有多次对比
                selected_split_idx_num = 1  # 参考data画图和以前的pdf数据吧
            # 平均选取
            selected_split_idx_num = min(selected_split_idx_num, cur_batch_size)
            selected_batch_idxes = np.linspace(0, cur_batch_size - 1, selected_split_idx_num, dtype=int) \
                if selected_split_idx_num != 0 else []
            # [45210] [45354] weather
            # FIXME: 补充画图的split_idx
            if mode == 'test':
                if data_name == 'Weather':
                    selected_batch_idxes = list(selected_batch_idxes)
                    if 45210 in split_idxes:
                        selected_batch_idxes.append(np.where(split_idxes == 45210)[0][0])
                    if 45354 in split_idxes:
                        selected_batch_idxes.append(np.where(split_idxes == 45354)[0][0])
                    selected_batch_idxes = np.array(selected_batch_idxes)
                if data_name == 'Traffic':  # [15832]
                    selected_batch_idxes = list(selected_batch_idxes)
                    if 15832 in split_idxes:
                        selected_batch_idxes.append(np.where(split_idxes == 15832)[0][0])
                    selected_batch_idxes = np.array(selected_batch_idxes)
            tr.time_end() if tr is not None else None
            for i in selected_batch_idxes:
                seq = historys[i, -3 * pred_len:, 0]
                pred = preds[i, :, 0]
                label = labels[i, :, 0]
                pred_line = np.concatenate((seq, pred), axis=0)
                label_line = np.concatenate((seq, label), axis=0)
                plt.figure(figsize=(ceil(len(label_line) / patch_len) * 5, 5))
                plt.plot(pred_line, label="pred", linestyle='--', color='blue')
                plt.plot(label_line, label="label", linestyle='-', color='orange')
                plt.legend()
                # 同一个样本的不同处理方式 方便比较
                plt.savefig(os.path.join(res_dir, mode, f'{split_idxes.tolist()[i]}-{index_str}.pdf'),
                            bbox_inches='tight')
            tr.time_start() if tr is not None else None
            if should_prune:
                break
        
        # FIXME：无论是否被prune，都要告诉tuner这个param_dict已经结束了，有点问题。。。mse； Ok:在筛选top1时会考虑max_step
        # FIXME: 这里的mean_mse在prune后不是在所有的split_idx上计算的！！！
        # FIXME：更高保真的mean_mse计算方式 -> 所有过去历史batch的全量split_idx的mse的mean
        # 不真实的mse->对TPE影响较大（单batch用batch_mse_list则不受坏影响
        if should_prune:
            final_mse = np.mean(batch_mse_list)
        else:
            final_mse = find_result_by_param_dict(pd_data, mode, params_space, param_dict)['mse']
        tuner.tell(param_dict, final_mse)
        logging.info(f"final_mse={final_mse}")
        terminate_flag = terminator_manager.update_and_check(final_mse, should_prune) \
            if terminator_manager is not None else False
        
        # # FIXME: 尝试用better_percent替代！！！！！！！！！intermediate怎么办... cur不好弄
        # better_percent_mse = calc_better_draw_percent(pd_data, mode, ['mse'], get_params_space_and_org()[1],
        #                                               param_dict)['better_percent_mse']
        # logging.info(f"better_percent_mse={better_percent_mse}")
        # tuner.tell(param_dict, better_percent_mse)
        # terminate_flag = terminator_manager.update_and_check(0, should_prune) \
        #     if terminator_manager is not None else False
        # terminate_flag = False
        
        if terminate_flag:
            logging.info(f"Experiment terminated by terminator!")
            break
    tr.time_end() if tr is not None else None
    t = time_start()
    pd_data.to_csv(os.path.join(res_dir, 'pd_data.csv'))
    grouped_cols = list(params_space.keys()) + ['mae', 'mse', 'rmse', 'mape', 'mspe'] + ['split_idx', 'idx']
    agg_dict = {'mae': 'mean', 'mse': 'mean', 'rmse': 'mean', 'mape': 'mean', 'mspe': 'mean',
                'split_idx': 'last', 'idx': 'last'}
    data_grouped_by_params = pd_data[pd_data['mode'] == mode][grouped_cols] \
        .groupby(list(params_space.keys())).agg(agg_dict).reset_index()
    # 没有区分prune，但是后续是pd_data再算top...
    data_grouped_by_params.to_csv(os.path.join(res_dir, f'_{mode}_data_grouped_by_params.csv'))
    log_time_delta(t, f"save {mode} data_grouped_by_params")
    tr.time_start() if tr is not None else None
    return pd_data


def main(data_name, model_name, target_column, pred_len, res_dir, device, args=None):
    model = get_model(model_name, device, args)
    dataset = get_dataset(data_name)

    pred_len = pred_len
    min_trimmer_seq_len = ceil(pred_len / patch_len) * patch_len
    
    # ? Add choose logic to select covariate use
    # params_space, origin_param_dict = get_params_space_and_org_cov()
    params_space, origin_param_dict = get_params_space_and_org()
    
    logging.info(f"params_space={params_space}")
    logging.info(f"origin_param_dict={origin_param_dict}")

    logging.info(f"params_space={params_space}")
    logging.info(f"origin_param_dict={origin_param_dict}")
    assert origin_param_dict.keys() == params_space.keys(), \
        f"origin_param_dict.keys()={origin_param_dict.keys()}, params_space.keys()={params_space.keys()}"
    # 确保org的str类型的参数值在values中
    for key in origin_param_dict.keys():
        if isinstance(origin_param_dict[key], str):
            assert origin_param_dict[key] in params_space[key]['values'], \
                f"origin_param_dict[{key}]={origin_param_dict[key]}, params_space[{key}]['values']={params_space[key]['values']}"
    # print("origin_param_dict", origin_param_dict)

    # max_seq_len = floor(dataset.train_len / 2)
    max_seq_len = params_space['trimmer_seq_len']['values'][-1]  # FIXME: 会改变split和norm等等

    for mode in ['train', 'test', 'val']:
        if not os.path.exists(os.path.join(res_dir, mode)):
            os.makedirs(os.path.join(res_dir, mode))

    # 改用pd.DataFrame存储结果数据
    # column: sampler_factor, trimmer_seq_len, aligner_method, normalizer_method, split_idx, mse!
    columns = list(params_space.keys()) + ['split_idx', 'mae', 'mse', 'rmse', 'mape', 'mspe'] + ['mode'] + ['idx']
    pd_data = pd.DataFrame(columns=columns)

    # FIXME：samples尽可能均衡 (问题：不同模型不平等 samples不同时间也不同。。。）  Ssm*Aug>600会不公平不定量！！！
    # __train_sample_num = get_max_batch_size_for_cuda(model_name)  # FIXME：这是对于rotate来说
    # __train_sample_num = 500  # 为什么1000会很差有异常值？？？hist画的也不对-> Ok:actual_batch_idx
    __train_sample_num = num_samples  # parser
    logging.info(f"__train_sample_num={__train_sample_num}")

    # 双min好像真不行...sample小影响泛化性筛选(5个貌似太少..??)...params小draw太大 (双min用来debug好用！
    train_data_factor = 1  # 数据取1/data_percent FIXME:样本数可以考虑从aug_mode=all补充！！！
    min_num_train_sample = __train_sample_num  # 10->直接看看val效果如何 100保证效果时间也还好
    max_train_sample_num = __train_sample_num  # 有cuda_batch_size限制了
    # max_num_params = 500  # 10->直接看看val效果如何 # 10就没有HPO什么事了。。。
    max_num_params = num_params  # parser
    logging.info(f"max_num_params={max_num_params}")
    # min_num_params_for_val = 0  # 很重要 75效果不错但是太多 10效果太差->平均119 # FIXME:其实不止1个还有pareto和org # 增强了pareto
    terminator_manager = TerminatorManager([  # FIXME
        # TimeLimitTerminator(60 * 10 if 'Chronos' in model_name else 60 * 5),
        MaxIterationsTerminator(max_num_params),
        # FIXME: 应该判断的是什么时候不只用100个！！！
        # NoImprovementTerminator('minimize', 0, 30),  # 25->40就停
        # RelativeImprovementTerminator('minimize', 0, 20, 40)  # 希望latest_best不要比previous_best差太多 # FIXME：探索难说
    ], 0, 0)  # FIXME: 100->67min 600->260min # 保护：至少要选出10个更好的！！！
    # FIXME: 用来排序的指标
    metric_weight_dict = {'mae': 2, 'mse': 3, 'rmse': 1, 'mape': 1, 'mspe': 1}  # 依据cumsum取值 -> 分3个level就行
    stats_weight_dict = {'mean': 3, 'std': 1, 'median': 2, 'iqr': 1, 'max': 1, 'min': 1}
    # metrics_for_pareto_must = ['mse', 'mae']  # 内部排列组合 以及 内部全+外部排列组合 -> Weather效果一般。。。
    # FIXME: pareto_statistics会排列组合数爆炸(4*5)！！！！
    # FIXME: 期望更鲁棒吧
    multi_pareto_mode = False
    metrics_for_pareto_must = ['mse']
    # multi_pareto_mode = True
    # metrics_for_pareto_must = list(metric_weight_dict.keys())  # 。。。
    # stats_for_pareto_must = list(stats_weight_dict.keys())  # 。。。
    # metrics_for_pareto_must = ['mse', 'mae']  # 。。。其实真差不多 可能多点更好更有多样性
    # metric_weight_dict_for_val = {k: v for k, v in metric_weight_dict.items() if k in ['mse', 'mae']}
    # stats_weight_dict_for_val = {k: v for k, v in stats_weight_dict.items() if k in ['mean']}
    # FIXME：多样性重要！??？（到什么程度？
    # metric_weight_dict_for_val = metric_weight_dict
    # stats_weight_dict_for_val = stats_weight_dict
    # metrics_for_pareto_must = ['mse', 'mae']  # 感觉还是不太行 得用all
    # metrics_for_pareto_must = list(metric_weight_dict.keys())  # 。。。 感觉一般。。。
    # stats_weight_dict_for_pareto = {k: v for k, v in stats_weight_dict.items() if k in ['mean']}  # all mean 还是不稳定？

    # metric_weight_dict_for_test = {k: v for k, v in metric_weight_dict.items() if k in ['mse', 'mae']}
    # metric_weight_dict_for_test = {k: v for k, v in metric_weight_dict.items() if k in ['mse']}
    metric_weight_dict_for_test = {k: v for k, v in metric_weight_dict.items() if k in ['mse', 'mae', 'rmse']}
    # FIXME：最后单看mean容易被击穿！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ 单median可能稳定性好点? 前提是pareto_must是[mse]
    stats_weight_dict_for_test = {k: v for k, v in stats_weight_dict.items() if k in ['mean', 'median']}
    # FIXME 纯用median不行？？？params100？
    # metric_weight_dict_for_test = metric_weight_dict  # FIXME:
    # stats_weight_dict_for_test = stats_weight_dict  # FIXME

    # FIXME: ablations
    logging.info(f"ablation={ablation}")
    # ablation = 'MultiMetric'
    metrics_for_pareto_must = ['mse'] if ablation == 'MultiMetric' else metrics_for_pareto_must
    metric_weight_dict_for_test = {'mse': 1} if ablation == 'MultiMetric' else metric_weight_dict_for_test
    # ablation = 'Stat'
    stats_weight_dict_for_test = {'mean': 1} if ablation == 'Stat' else stats_weight_dict_for_test
    # ablation = 'MultiPareto'
    # multi_pareto_mode = False if ablation == 'MultiPareto' else multi_pareto_mode  # pareto_must_mode开启则代表 选择多个多目标优化的topk(每个多目标优化)

    # # FIXME: prune的相关参数 !!!!!!单样本mse真不至于太严格，还是用多指标筛选更合理！其实是多样本prune还好 100->基本剩80
    # train_prune_percentile = 75
    # # train_prune_step = np.inf  # inf还是谈严格了导致不必要的draw。。。      rotate还是all？没事，都是none整倍数
    # # 只针对ground的prune！！！不成倍数也太严格了（不同样本差异会大
    # interval_step = len(Augmentor('none', 'all', pred_len).aug_method_dict.keys()) * 1  # FIXME: 越小越严格！

    tr_adapt = TimeRecorder('adapt_duration')
    tr_adapt.time_start()
    # ########################################################### train
    mode = 'train'
    total_num_train_sample = len(dataset.get_available_idx_list(mode, max_seq_len, pred_len))
    logging.info(f"total_num_train_sample={total_num_train_sample}")
    # 生成所有参数组合
    # params_list = itertools.product(*[params_space[key]['values'] for key in params_space.keys()])
    # param_dict_list = [dict(zip(params_space.keys(), params)) for params in params_list]
    # 训练集样本数

    calculated_sample_num = ceil(total_num_train_sample / train_data_factor) if not fast_mode else 10
    num_train_sample = max(min(calculated_sample_num, max_train_sample_num), min_num_train_sample)
    # num_train_sample = min(num_train_sample, 30) if 'Chronos' in model_name else num_train_sample  # FIXME: chronos OOM
    # FIXME: aug=all 已经不是真实的batch_size了。。。
    train_batch_sizes = [ceil(num_train_sample / 1)] if not fast_mode else [num_train_sample]  # FIXME:# 数据长度!=1太慢！
    logging.info(f"num_train_sample={num_train_sample}, train_batch_sizes={train_batch_sizes}")
    # max_train_duration = 60 * 60 * 3  # 3h
    train_aug_method, train_aug_mode = 'none', 'rotate'  # FIXME

    train_aug_mode = 'fix' if ablation == 'Aug' else train_aug_mode
    train_aug_mode = 'fix' if ablation == 'Val' else train_aug_mode

    train_augmentor = Augmentor(train_aug_method, train_aug_mode, pred_len)
    aug_batch_size = int(len(train_augmentor.aug_method_dict.keys()))
    real_train_batch_size = max(get_max_batch_size_for_cuda(model_name), aug_batch_size * train_batch_sizes[0])
    logging.info(f"max_num_params={max_num_params}")
    # train_pruner_kwargs = {'percentile': train_prune_percentile, 'n_startup_trials': 1, 'n_warmup_steps': 0,
    #                        # 'interval_steps': max(ceil(real_train_batch_size / train_prune_step), 1),
    #                        'interval_steps': interval_step,
    #                        # Fixme: prune太多？？？ceil(real_train_batch_size / train_prune_step)
    #                        'n_min_trials': 10}  # FIXME (1/2)**7=128 n_min_trials保底... max_step=3或5不错不要多 25-5:100->15 n_min_trials=1在小params时很重要!!!！
    # train_tuner = OptunaTuner(params_space, 'minimize', 'TPESampler', 'PercentilePruner', train_pruner_kwargs,
    #                           [origin_param_dict])  # FIXME：!!!!!!! RandomSampler？TPESampler在100以上不错
    train_tuner = 'TPESampler'
    train_tuner = 'RandomSampler' if ablation == 'HPO' else train_tuner
    train_tuner = OptunaTuner(params_space, 'maximize', train_tuner, 'NoPruner', None, [origin_param_dict])
    # train_tuner = OptunaTuner(params_space, 'maximize', 'RandomSampler', 'NoPruner', None, [origin_param_dict])

    train_pruner_report_mode = 'single'  # 'single', 'batch' # FIXME
    train_pruner_metric_mode = 'cur'  # 'mean', 'cur' # FIXME
    # 开始训练！（适配）
    # ? Add choose logic to select covariate using
    # pd_data = train_or_test_or_val_cov(patch_len, pred_len, max_seq_len,
    #                                model, dataset, target_column, mode, num_train_sample, train_augmentor,
    #                                train_batch_sizes,
    #                                params_space, train_tuner, train_pruner_report_mode, train_pruner_metric_mode,
    #                                max_num_params, terminator_manager,
    #                                pd_data, res_dir, tr_adapt)
    pd_data = train_or_test_or_val(patch_len, pred_len, max_seq_len,
                                   model, dataset, target_column, mode, num_train_sample, train_augmentor,
                                   train_batch_sizes,
                                   params_space, train_tuner, train_pruner_report_mode, train_pruner_metric_mode,
                                   max_num_params, terminator_manager,
                                   pd_data, res_dir, tr_adapt, args)
    tr_adapt.time_end()
    many_plot(pd_data, mode, res_dir)
    tr_adapt.time_start()
    # 计算实际参数数量
    actual_num_params = max_num_params
    for terminator in terminator_manager.terminators:
        if isinstance(terminator, MaxIterationsTerminator):
            actual_num_params = terminator.iteration_count
    logging.info(f"actual_num_params={actual_num_params}")

    # ########################################################### valid
    # FIXME：1010测试！说明val=num_train_sample且no-aug时不错！
    mode = 'val'  # FIXME:虽然是val模式但是使用的是train的数据！！！！
    # total_num_val_sample = len(dataset.get_available_idx_list(mode, max_seq_len, pred_len))
    # logging.info(f"total_num_val_sample={total_num_val_sample}")
    #     num_val_sample = min(ceil(total_num_val_sample / data_percent), max_sample_num) if not fast_mode else 2
    # num_val_sample = ceil(num_train_sample / 10) if not fast_mode else 2  # FIXME: 变相降低val的样本数
    num_val_sample = num_train_sample if not fast_mode else 2  # FIXME：增加val_sample试试-》多了不好除非有prune！！！
    # num_val_sample = min(num_val_sample, 30) if 'Chronos' in model_name else num_train_sample  # FIXME: chronos OOM
    # 就10个组合没必要prune也就没必要多batch -> 不为效率为鲁棒性？？！！->在不同部分的数据上都达到前percentile！！！！！！
    val_batch_sizes = [ceil(num_val_sample / 1)] if not fast_mode else [num_val_sample]  # FIXME: (1/2)**3=1/8
    # val_batch_sizes = [100] if not fast_mode else [num_val_sample]  # FIXME:
    logging.info(f"num_val_sample={num_val_sample}, val_batch_sizes={val_batch_sizes}")
    # 为了避免过拟合的算子：!!! 额外添加org和pareto_dominance到列表中（但是感觉pareto没什么用...）
    # metrics_for_pareto_dominance = ['rmse', 'mape', 'mspe'] if not fast_mode else ['rmse']  # FIXME:
    # 生成所有可能的 metric 组合，长度从 1 到 metrics_for_pareto_dominance 的长度
    # pareto_top_k = 5  # FIXME:多了会好一点点的样子...
    pareto_top_k = 3  # FIXME: 防止差的mse漏网之鱼
    pareto_top_k = max(1, floor(actual_num_params / 30))
    # pareto_top_k = ceil(max_num_params // 20)  # 多给一点试试
    # FIXME: 是否考虑statistics综合排名？No -> pareto_statistics会排列组合数爆炸 -> 考虑不使用pareto???那多样性就少了
    pareto_param_dict_list = find_top_param_dict_list_pareto(pd_data, 'train', params_space, pareto_top_k,
                                                             metric_weight_dict, metrics_for_pareto_must,
                                                             multi_pareto_mode,
                                                             res_dir)
    # pareto_param_dict_list = find_top_param_dict_list_pareto_by_statistics(
    #     pd_data, 'train', params_space, top_k, metric_weight_dict, stats_weight_dict, metrics_for_pareto_must,
    #     stats_for_pareto_must, res_dir)
    # pareto_param_dict_list = find_top_param_dict_list_by_statistics(
    #     pd_data, 'train', params_space, top_k * 3, metric_weight_dict, stats_weight_dict, res_dir)
    # fixme:尝试统一的筛选方法？？？ 效果确实一般 多样性感觉不行。。。？？？？多样性排名干扰有点大 aug上的mse太差
    # pareto_param_dict_list = find_top_param_dict_list_by_statistics(pd_data, 'train', params_space, pareto_top_k,
    #                                                                 metric_weight_dict_for_val,
    #                                                                 stats_weight_dict_for_val,
    #                                                                 res_dir)
    # 更新后的pareto_by_statistics，metric数量不会爆炸了
    # pareto_param_dict_list = find_top_param_dict_list_pareto_by_statistics(pd_data, 'train', params_space, pareto_top_k,
    #                                                                        metric_weight_dict, metrics_for_pareto_must,
    #                                                                        stats_weight_dict_for_pareto, res_dir)

    # 合并 param_dict_list_for_val # FIXME:考虑不添加origin_param_dict以减少draw!!!（有org才能画分布图。。
    val_unique_param_dict_list = make_param_dict_unique([origin_param_dict] + pareto_param_dict_list)
    # val_unique_param_dict_list = make_param_dict_unique(pareto_param_dict_list)
    logging.info(f"len(val_unique_param_dict_list): {len(val_unique_param_dict_list)}")
    logging.debug(f"val_unique_param_dict_list: {val_unique_param_dict_list}")
    val_pruner_kwargs = None  # FIXME: 真实场景比较效果好！（Uni2ts）# FIXME:NopPruner
    val_tuner = OptunaTuner(params_space, 'minimize', 'TPESampler', 'NoPruner', val_pruner_kwargs,
                            val_unique_param_dict_list)
    # val_pruner_kwargs = {'percentile': 25, 'n_startup_trials': 1, 'n_warmup_steps': 0,
    #                      'interval_steps': train_batch_sizes[0] // 5,
    #                      'n_min_trials': 1}
    # val_tuner = OptunaTuner(params_space, 'minimize', 'TPESampler', 'PercentilePruner', val_pruner_kwargs,
    #                         val_unique_param_dict_list)
    val_pruner_report_mode = 'single'  # 'single', 'batch' # FIXME
    val_pruner_metric_mode = 'cur'  # 'mean', 'cur' # FIXME
    # FIXME: rotate=False期望真实场景比较 # True对Uni2ts的效果？很一般 但是pareto留下了一些差的。。。很需要 # pareto已修复！？？？
    val_augmentor = Augmentor('none', 'fix', pred_len)
    # 开始验证！
    pd_data = train_or_test_or_val(patch_len, pred_len, max_seq_len,
                                   model, dataset, target_column, mode, num_val_sample, val_augmentor,
                                   val_batch_sizes,
                                   params_space, val_tuner, val_pruner_report_mode, val_pruner_metric_mode,
                                   len(val_unique_param_dict_list), None,
                                   pd_data, res_dir, tr_adapt, args=args)
    tr_adapt.time_end()
    many_plot(pd_data, mode, res_dir)

    tr_test = TimeRecorder('test_duration')
    tr_test.time_start()
    # ########################################################### test
    mode = 'test'
    torch.cuda.empty_cache()
    # 从val中找top_k_for_test的参数组合
    # FIXME: 是否考虑statistics综合排名？ ！！！！！当然，只看mean不行还要看median！
    # val_top1_param_dict = find_top_param_dict_list(pd_data, 'val', params_space, 1, metric_weight_dict_for_test,
    #                                                res_dir)[0]
    val_top1_param_dict = find_top_param_dict_list_by_statistics(pd_data, 'val', params_space, 1,
                                                                 metric_weight_dict_for_test,
                                                                 stats_weight_dict_for_test,
                                                                 res_dir)[0]
    # FIXME: train top1 的综合排序 试试！！！
    # val_top1_param_dict = find_top_param_dict_list(pd_data, 'train', params_space, 1, metric_weight_dict_for_test,
    #                                                res_dir)[0]
    # val_top1_param_dict = find_top_param_dict_list_by_statistics(pd_data, 'train', params_space, 1,
    #                                                              metric_weight_dict_for_test,
    #                                                              stats_weight_dict_for_test,
    #                                                              res_dir)[0]
    if ablation == 'Val':  # 使用train的top1
        # 太差了不合适
        val_top1_param_dict = find_top_param_dict_list(pd_data, 'train', params_space, 1, {'mse': 1}, res_dir)[0]
        # val_top1_param_dict = find_top_param_dict_list(pd_data, 'train', params_space, 1, metric_weight_dict_for_test,
        #                                                res_dir)[0]

    logging.info(f"val_top1_param_dict={val_top1_param_dict}")#! This is the Best Preprocess Technique
    # if 'Timer' in model_name:
    #     from Timer.exp.exp_large_few_shot_roll_demo import Exp_Large_Few_Shot_Roll_Demo
    #     Exp = Exp_Large_Few_Shot_Roll_Demo
    #     from pipeline import Process
    #     process = Process(model, dataset, val_top1_param_dict)
    #     # setting record of experiments
    #     print('Args in experiment:')
    #     print(args)
    #     if args.is_finetuning:
    #         for ii in range(args.itr):
    #             # setting record of experiments
    #             setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_predl{}_patchl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
    #                 args.task_name,
    #                 args.model_id,
    #                 args.model,
    #                 args.data,
    #                 args.features,
    #                 args.seq_len,
    #                 args.label_len,
    #                 args.pred_len,
    #                 args.patch_len,
    #                 args.d_model,
    #                 args.n_heads,
    #                 args.e_layers,
    #                 args.d_layers,
    #                 args.d_ff,
    #                 args.factor,
    #                 args.embed,
    #                 args.distil,
    #                 args.des,
    #                 ii)
    #             if args.date_record:
    #                 # setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    #                 setting = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S") + setting
    #             
    #             exp = Exp(args)  # set experiments
    #             print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    #             exp.finetune(setting, process)
    #             
    #             print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #             exp.test(setting, process)
    #             torch.cuda.empty_cache()
    #     else:
    #         ii = 0
    #         setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
    #             args.task_name,
    #             args.model_id,
    #             args.model,
    #             args.data,
    #             args.features,
    #             args.seq_len,
    #             args.label_len,
    #             args.pred_len,
    #             args.d_model,
    #             args.n_heads,
    #             args.e_layers,
    #             args.d_layers,
    #             args.d_ff,
    #             args.factor,
    #             args.embed,
    #             args.distil,
    #             args.des,
    #             ii)
    #         if args.date_record:
    #             # setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    #             setting = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S") + setting
    #         exp = Exp(args)  # set experiments
    #         print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #         exp.test(setting,process, test=1)
    #         torch.cuda.empty_cache()

    # 添加原始参数到列表中
    # test_unique_param_dict_list = make_param_dict_unique([origin_param_dict, val_top1_param_dict])
    # test_unique_param_dict_list = make_param_dict_unique([origin_param_dict, val_top1_param_dict, train_top1_param_dict])  # 方便观察最终筛选
    test_unique_param_dict_list = make_param_dict_unique([origin_param_dict, val_top1_param_dict])  # 方便观察
    logging.info(f"test_unique_param_dict_list={test_unique_param_dict_list}")
    # 测试集
    num_test_sample = 'all' if not fast_mode else 2  # !!!!!!!! FIXME: 为了快速测试
    test_batch_sizes = [get_max_batch_size_for_cuda(model_name)] if not fast_mode else [100]  # FIXME: 还是容易崩 Ok
    if model_name=='Arima':
        test_batch_sizes = [100]
    # test_batch_sizes = [min(test_batch_sizes[0], 30)] if 'Chronos' in model_name else test_batch_sizes
    logging.info(f"num_test_sample={num_test_sample}, test_batch_sizes={test_batch_sizes}")
    # 遍历param_dict_list，计算结果
    test_tuner = MyBatchTuner(params_space, test_unique_param_dict_list)
    test_augmentor = Augmentor('none', 'fix', pred_len)
    # 不prune 但是方便记录和debug个别异常
    test_pruner_report_mode = 'single'  # 'single', 'batch' # FIXME
    test_pruner_metric_mode = 'cur'  # 'mean', 'cur' # FIXME
    # 开始测试！（适配）
    pd_data = train_or_test_or_val(patch_len, pred_len, max_seq_len,
                                   model, dataset, target_column, mode, num_test_sample, test_augmentor,
                                   test_batch_sizes,
                                   params_space, test_tuner, test_pruner_report_mode, test_pruner_metric_mode,
                                   len(test_unique_param_dict_list), None,
                                   pd_data, res_dir, tr_test, args=args)
    tr_test.time_end()
    many_plot(pd_data, mode, res_dir)
    # 画直方图直观比较org和our的metric的分布差异
    plot_metric_hist_comparison(pd_data, 'test', metric_names, res_dir, val_top1_param_dict)
    plot_metric_hist_comparison(pd_data, 'val', metric_names, res_dir, val_top1_param_dict) \
        if ablation != 'Val' else None
    plot_metric_hist_comparison(pd_data, 'train', metric_names, res_dir, val_top1_param_dict)  # train也画！

    res_dict = {}
    # 将org和our在test上的mse等指标返回（mean-based）先求得不同samples的MSE的mean，再用相对提升率的公式计算
    res_dict.update(calc_improve_percent1(pd_data, 'test', params_space, metric_names, val_top1_param_dict))
    # 将org和our在test上的mse等指标返回（sample-based）直接求得不同samples对应的MSE的相对提升率再mean
    res_dict.update(calc_improve_percent2(pd_data, 'test', metric_names, val_top1_param_dict))
    # 计算mean-based的 mean, median, iqr, std 的提升率
    res_dict.update(calc_improve_percent_statistics(pd_data, 'test', metric_names, val_top1_param_dict))

    # FIXME: 下面三个基本被废弃
    # 计算val_top1_param_dict和org在test上不同sample的某metric占优的比率
    res_dict.update(calc_better_draw_percent(pd_data, 'test', metric_names, val_top1_param_dict))
    # 计算val_top1_param_dict和org在test上Bett的sample和Bad的sample分别提升和降低的比率
    res_dict.update(calc_improve_percent_in_better_and_worse(pd_data, 'test', metric_names, val_top1_param_dict))
    # 计算val_top1_param_dict和org在test上hard, medium, easy的sample中的提升比率
    res_dict.update(calc_improve_percent_in_hard_medium_easy(pd_data, 'test', metric_names, val_top1_param_dict))

    # 在结果中加入params数量和val的top1的params超参数取值
    res_dict[f'num_params'] = actual_num_params
    for key, value in val_top1_param_dict.items():
        res_dict[key] = value
    adapt_duration, test_duration = tr_adapt.get_total_duration(), tr_test.get_total_duration()
    res_dict['adapt_duration'] = adapt_duration
    res_dict['test_duration'] = test_duration
    res_dict['adapt_duration_percent'] = adapt_duration / (adapt_duration + test_duration) * 100
    logging.info(f"res_dict={res_dict}")
    return res_dict


if __name__ == "__main__":
    set_seed(seed)

    if fast_mode:
        profiler = cProfile.Profile()
        profiler.enable()
    gpu_index=args.gpu
    date_time_str = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    setting_suffix = f"-P{num_params}-S{num_samples}-AB{ablation}-SD{seed}-MP{max_processes}"
    res_root_dir = os.path.join('new_moti' if not fast_mode else 'debug', date_time_str + setting_suffix)
    os.makedirs(f'{res_root_dir}', exist_ok=True)
    
    print(f"data_name={data_name}, model_name={model_name}, pred_len={pred_len}, "
          f"use_gpu={use_gpu}, gpu_indexes={args.gpu}")
    res_dir = os.path.join(res_root_dir, data_name, model_name, f'pred_len-{pred_len}')
    atexit.register(atexit_handler, res_dir)
    # device = f"cuda:{gpu_indexes.split(',')[0]}" if use_gpu else 'cpu'
    device = f"cuda:{args.gpu}" if use_gpu else 'cpu'
    print(f"res_dir={res_dir}, device={device}")

    # log_file = os.path.join(res_dir, 'exp_single.log')
    logger = logging.getLogger()
    for handler in logger.handlers[:]:  # 删除很重要！之前multi
        logger.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    logging.info(f"num_samples={num_samples}, num_params={num_params}, ablation={ablation}, seed={seed}")

    # if 'ETT' in data_name:
    #     column_names = EttHour.column_names if not fast_mode else ['HUFL', 'OT']
    # elif 'Exchange' == data_name:
    #     column_names = Exchange.column_names if not fast_mode else ['0', 'OT']
    # else:
    #     raise ValueError(f"Unknown data_name: {data_name}")
    # FIXME: 现在都只看OT了
    column_names = ['OT']

    cmp_method = ['org', 'our',
                  'improve_percent1',
                  'improve_percent2',
                  'improve_percent_median', 'improve_percent_iqr',
                  'improve_percent_mean', 'improve_percent_std',
                  'improve_percent_max', 'improve_percent_min',

                  'better_percent', 'draw_percent',
                  'improve_percent_in_better', 'improve_percent_in_worse',
                  'improve_percent_in_hard', 'improve_percent_in_medium', 'improve_percent_in_easy'
                  ]
    metric_names = ['mae', 'mse', 'rmse', 'mape', 'mspe']
    cmp_metric_columns = [f"{cmp}_{metric}" for metric in metric_names for cmp in cmp_method]
    basic_column = ['data_name', 'model_name', 'pred_len']
    params_columns = ['num_params'] + list(get_params_space_and_org()[0].keys())
    duration_columns = ['adapt_duration', 'test_duration', 'adapt_duration_percent']
    detailed_results = pd.DataFrame(columns=basic_column + ['target_column'] + cmp_metric_columns + params_columns +
                                            duration_columns)
    t = time_start()
    for target_column in column_names:
        logging.info('###############################################')
        logging.info(f"Begin to process {data_name} {model_name} {target_column}...")
        res_dir_for_column = os.path.join(res_dir, target_column)
        result_dict = main(data_name, model_name, target_column, pred_len, res_dir_for_column, device, args=args)
        record = {**result_dict, 'data_name': data_name, 'model_name': model_name,
                  'target_column': target_column, 'pred_len': pred_len}
        detailed_results.loc[len(detailed_results)] = record
        # 每周期都写入数据防止丢失
        detailed_results.to_csv(os.path.join(res_dir, 'detailed_results.csv'))
        logging.info(f"detailed_results=\n{detailed_results.to_string()}")

        # 每周期都写入数据防止丢失
        # 汇总结果, 按照basic_column分组，实际是把不同的target_column的metrics用均值聚合了（不管params了）
        # 如果target_column只有一个，那就也罢params的都加上！
        if len(column_names) == 1:
            summary_results = detailed_results
        else:
            summary_results = detailed_results[basic_column + cmp_metric_columns + ['num_params'] + duration_columns] \
                .groupby(basic_column).mean().reset_index()
        summary_results.to_csv(os.path.join(res_dir, 'summary_results.csv'))
        logging.info(f"summary_results=\n{summary_results.to_string()}")
    log_time_delta(t, 'all')

    if fast_mode:  # FIXME
        # Your code here
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(10)  # Print the top 10 time-consuming functions
