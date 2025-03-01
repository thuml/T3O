import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import DataLoader

import matplotlib
import torch.cuda.amp as amp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import get_dataset, CustomDataset
from AnyTransform.model import get_model
from AnyTransform.tuners import OptunaTuner
from AnyTransform.pipeline import adaptive_infer
from AnyTransform.utils import TimeRecorder, get_params_space_and_org


def run(model, dataset, custom_dataset, batch_size, param_dict):
    process_dur, model_dur = 0, 0
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
    bar2 = enumerate(dataloader)
    count = 0
    for step_idx, (split_idxes, aug_methods, historys, labels) in bar2:
        logging.info(f"{split_idxes.shape=}, {historys.shape=}, {labels.shape=}")
        assert len(historys.shape) == 4 and len(labels.shape) == 4, \
            f"historys.shape={historys.shape}, labels.shape={labels.shape}"
        logging.info(f"{step_idx=}")
        # 转化成3维
        historys = historys.reshape(-1, historys.shape[2], historys.shape[3])
        labels = labels.reshape(-1, labels.shape[2], labels.shape[3])
        # 转化成numpy
        historys, labels = historys.numpy(), labels.numpy()

        # 用 adaptive_infer 推理
        kwargs = param_dict.copy()
        kwargs.update({'history_seqs': historys, 'model': model, 'dataset': dataset,
                       'target_column': 'OT', 'patch_len': 96, 'pred_len': 96, 'mode': 'test'})
        # 开始计时
        preds, _process_dur, _model_dur = adaptive_infer(**kwargs)
        logging.info(f"{process_dur=}, {model_dur=}")
        process_dur += _process_dur
        model_dur += _model_dur
        count += 1
    process_dur /= count
    model_dur /= count
    return process_dur, model_dur


def get_dur(model, dataset, custom_dataset, batch_size, num_sample):
    logging.info(f"get_dur {model=}")
    # 随便从ETTh1搞10段数据，预测96长度，记录时间
    batch_num = num_sample // batch_size
    params_space, origin_param_dict = get_params_space_and_org(fast_mode=False)
    our_param_dict = OptunaTuner(params_space, 'minimize', 'RandomSampler', 'NoPruner', None, None).ask()

    logging.info(f"Begin {origin_param_dict=}, {our_param_dict=}")
    # org_step_duration = run(model, dataset, custom_dataset, batch_size, origin_param_dict)
    # our_step_duration = run(model, dataset, custom_dataset, batch_size, our_param_dict)

    # 方案1
    process_dur, model_dur = run(model, dataset, custom_dataset, batch_size, our_param_dict)

    # # 方案2：随机性较大
    # _, _ = run(model, dataset, custom_dataset, batch_size, origin_param_dict)  # 预热
    # org_process_dur, org_model_dur = run(model, dataset, custom_dataset, batch_size, origin_param_dict)
    # our_process_dur, our_model_dur = run(model, dataset, custom_dataset, batch_size, our_param_dict)
    # org_dur = org_process_dur + org_model_dur
    # our_dur = our_process_dur + our_model_dur
    # model_dur = org_dur
    # process_dur = our_dur - org_dur  # 是可能比原来快的 因为patch和sample让seq变小了

    return process_dur, model_dur


def main():
    # 单进程！！！
    # 测试推理阶段的额外消耗 params和samples等适配阶段的参数不重要 data本身也不太重要因为只看时间
    # model_names = ['Timer-UTSD', 'MOIRAI-base', 'Chronos-tiny']
    model_names = ['Timer-UTSD', 'MOIRAI-base', 'Chronos-tiny']  # FIXME
    logging.info(f"{model_names=}")
    # batch_sizes = [1, 25, 100, 500] # 10	25	50	100
    batch_sizes = [1, 5, 10, 25] + [50, 100]  # FIXME: 推理的时候短些合理 50,100留给适配阶段可能需要的计算
    logging.info(f"{batch_sizes=}")
    df_columns = ['model_name', 'batch_size', 'process_dur', 'model_dur', 'process_model_ratio', 'repeat']

    seq_len = 96 * 15
    pred_len = 96
    logging.info(f"{seq_len=}, {pred_len=}")
    dataset = get_dataset('ETTh1')
    mode, target_column, max_seq_len, augmentor = \
        'test', 'OT', seq_len, Augmentor('none', 'fix', pred_len)

    repeats = [1, 2, 3, 4, 5]
    logging.info(f"{repeats=}")
    num_batch = 5
    logging.info(f"{num_batch=}")
    df_data = pd.DataFrame(columns=df_columns)

    # repeats = [1, 2, 3]
    for repeat in repeats:
        logging.info(f"########################################{repeat=}")
        for batch_size in batch_sizes:
            logging.info(f"########################################{batch_size=}")
            num_sample = num_batch * batch_size
            custom_dataset = CustomDataset(dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample)
            for model_name in model_names:
                logging.info(f"########################################{model_name=}")
                model = get_model(model_name, 'cuda:0' if torch.cuda.is_available() else 'cpu')
                process_dur, model_dur = get_dur(model, dataset, custom_dataset, batch_size, num_sample)
                process_model_r = process_dur / model_dur
                logging.info(f"{model_name=}, {process_dur=}, {model_dur=}")
                df_data.loc[len(df_data)] = [model_name, batch_size, process_dur, model_dur, process_model_r, repeat]

                # 保存
                df_data.to_csv(os.path.join(res_dir, '_experiment_duration.csv'), index=False)


if __name__ == "__main__":
    # cd ~/PycharmProjects/ts_adaptive_inference && python3 AnyTransform/exp_duration.py --data_name 0 --model_name 0 --pred_len 0 --res_root_dir 0
    tr = TimeRecorder('exp_duration')
    tr.time_start()
    date_time_str = datetime.now().strftime("%y%m%d-%H%M%S")
    res_dir = os.path.join('new_moti', date_time_str + '-duration')
    os.makedirs(res_dir, exist_ok=True)

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    log_file = os.path.join(res_dir, '_duration_exp.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler(log_file, 'w', 'utf-8'), logging.StreamHandler(sys.stdout)])
    logging.info(f"res_dir={res_dir}")
    main()
    tr.time_end()
    d = tr.get_total_duration()
    logging.info(f"Total duration: {d}")
