import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True
    return False


matplotlib.use('TkAgg') if is_pycharm() else None


def trans_time_str(s):
    tokens = s.split('-')
    time_str = '-'.join([tokens[0], tokens[1]])
    time_format = "%y%m%d-%H%M%S"
    t = datetime.strptime(time_str, time_format)
    return t


def get_dirs_from_range(parent_dir, start, end):
    child_dirs = []
    time_start = trans_time_str(start)
    time_end = trans_time_str(end) if end is not None else datetime.now()

    for child_dir in os.listdir(parent_dir):
        tokens = child_dir.split('-')
        if len(tokens) < 6:
            continue

        time_cur = trans_time_str(child_dir)
        if time_start <= time_cur <= time_end:
            child_dirs.append(child_dir)
    # sort
    child_dirs.sort()
    print('child_dirs', child_dirs)
    return child_dirs


def generate_latex_table(df_samples, df_params, df_duration, models, file_name='latex_table_duration.tex'):
    # 参考
    # \multirow{3}{*}{Trials}
    # & 25 & 2.99 s & 5.48 s & 29.14 s \\
    # & 100 & 11.21 s & 18.05 s & 112.39 s \\
    # & 500 & 75.81 s & 104.18 s & 578.90 s \\
    # \midrule
    # \multirow{3}{*}{Samples}
    # & 25 & 6.80 s & 11.39 s & 87.71 s \\
    # & 100 & 11.21 s & 18.05 s & 112.39 s \\
    # & 500 & 37.44 s & 65.90 s & 398.03 s \\
    with open(file_name, 'w') as f:
        f.write('\\multirow{3}{*}{Trials}\n')

        # Adaptation Overhead by Number of Operator Combinations
        # params_nums = [50, 100, 500]
        params_nums = list(df_params['params'].unique())
        for params in params_nums:
            f.write(f'& {params}') if params_nums.index(params) != 0 else f.write(f'& {params}')
            for model in models:
                adapt_duration = df_params[(df_params['params'] == params) & (df_params['model'] == model)][
                    'adapt_duration'].mean()
                f.write(f' & {adapt_duration:.2f} s')
            f.write(' \\\\\n')

        f.write('\\midrule\n')
        f.write('\\multirow{3}{*}{Samples}\n')

        # Adaptation Overhead by Number of Data Samples
        # samples_nums = [50, 100, 500]
        samples_nums = list(df_samples['samples'].unique())
        for samples in samples_nums:
            f.write(f'& {samples}') if samples_nums.index(samples) != 0 else f.write(f'& {samples}')
            for model in models:
                adapt_duration = df_samples[(df_samples['samples'] == samples) & (df_samples['model'] == model)][
                    'adapt_duration'].mean()
                f.write(f' & {adapt_duration:.2f} s')
            f.write(' \\\\\n')

        # f.write('\\cmidrule(lr){1-6}\n')
        # f.write(
        #     '\\multirow{3}{*}{\\shortstack{Inference Overhead \\\\ (per prediction)}} & \\multirow{3}{*}{\\shortstack{Inference \\\\ Batch Size}}\n')
        #
        # # Inference Overhead by Inference Batch Size and Model
        # # batch_sizes = [1, 5, 10, 25, 50]
        # batch_sizes = [1, 10, 100]
        # for batch_size in batch_sizes:
        #     f.write(f'& & {batch_size}') if batch_sizes.index(batch_size) != 0 else f.write(f'& {batch_size}')
        #     for model in models:
        #         infer_duration = df_duration[(df_duration['batch_size'] == batch_size) & (
        #                 df_duration['model_name'] == model)]['process_dur'].mean()
        #         print(f"{model=}, {batch_size=}, {infer_duration=}")
        #         f.write(f' & {infer_duration * 1000:.2f} ms')
        #     f.write(' \\\\\n')

        # f.write('\\bottomrule\n')
        # f.write('\\end{tabular}\n')
        # f.write('\\end{table}\n')


def main(parent_dir_name, children_dir_names):
    columns = [
        'samples', 'params', 'seed', 'model', 'adapt_duration'
    ]

    df_data = pd.DataFrame(columns=columns)

    pred_len = 96
    data = 'ETTh1'
    models = ['Timer-UTSD', 'MOIRAI-base', 'Chronos-tiny']
    fix_num_sample = 100
    fix_num_param = 100
    sample_nums = [25, 100, 500]
    param_nums = [25, 100, 500]  # 好像就这只有这三个？？？
    valid_children_dir_names = []
    for child_dir_name in children_dir_names:
        tokens = child_dir_name.split('-')
        if len(tokens) < 6:
            continue
        ablation = [token[2:] for token in tokens if token.startswith('AB')][0]
        num_param = [int(token[1:]) for token in tokens if token.startswith('P')][0]
        num_sample = [int(token[1:]) for token in tokens if token.startswith('S') and not token.startswith('SD')][0]
        seed = [int(token[2:]) for token in tokens if token.startswith('SD')][0]

        if ablation != 'none':
            continue
        if not ((num_sample == fix_num_sample and num_param in param_nums) or \
                (num_sample in sample_nums and num_param == fix_num_param)):
            continue
        print(f"{num_sample=}, {num_param=}, {ablation=}, {seed=}")
        valid_children_dir_names.append(child_dir_name)

        # file_path = os.path.join(parent_dir_name, child_dir_name, '_exp_summary_total.csv')
        file_path = os.path.join(parent_dir_name, child_dir_name, '_experiment_status.csv')
        df_exp = pd.read_csv(file_path)
        # print(df_exp)
        # data和predlen是固定的
        df_exp_filtered = df_exp[df_exp['data_name'] == data]
        df_exp_filtered = df_exp_filtered[df_exp_filtered['pred_len'] == pred_len]
        # data_name	model_name	pred_len	org_mae	our_mae	improve_percent1_mae
        # ETTh1	Timer-UTSD	96	0.574760445	0.552896707	3.803974023
        # ETTh1	MOIRAI-base	96	0.560801533	0.52480084	6.419506813
        # ETTh1	Chronos-tiny	96	0.544198715	0.524907075	3.544962325
        # ETTh1	Timer-LOTSA	96	0.52537805	0.52537805	0
        # ETTh1	MOIRAI-small	96	0.546375891	0.525094672	3.894977756
        # ETTh1	MOIRAI-large	96	0.536430892	0.525258833	2.082665107

        # 每个model添加一条：
        for model in models:
            df_data.loc[len(df_data)] = [
                num_sample, num_param, seed,
                model, df_exp_filtered[df_exp_filtered['model_name'] == model]['adapt_duration'].values[0]
            ]

    print("len(valid_children_dir_names):", len(valid_children_dir_names))
    print("valid_children_dir_names:", valid_children_dir_names)

    df_data = df_data.drop(columns=['seed'])
    print(df_data)

    df_data_fix_num_params = df_data[df_data['params'] == fix_num_param]
    df_data_fix_params_mean = df_data_fix_num_params.groupby(['samples', 'model']).mean().reset_index()
    print(f'df_data_fix_params_mean=\n{df_data_fix_params_mean}')

    df_data_fix_num_samples = df_data[df_data['samples'] == fix_num_sample]
    df_data_fix_samples_mean = df_data_fix_num_samples.groupby(['params', 'model']).mean().reset_index()
    print(f'df_data_fix_samples_mean=\n{df_data_fix_samples_mean}')

    duration_dir = '240715-002025-duration-more-num!!!!!!!!'
    duration_df = pd.read_csv(os.path.join(parent_dir_name, duration_dir, '_experiment_duration.csv'))
    # model_name	batch_size	process_dur	model_dur	process_model_ratio	repeat
    # Timer-UTSD	1	0.003323221	0.061053514	0.054431284	1
    # MOIRAI-base	1	0.002137899	0.081508923	0.026229023	1
    # Chronos-tiny	1	0.002355576	0.763283539	0.003086108	1
    # Timer-UTSD	25	0.008451796	0.027231693	0.310366142	1
    # MOIRAI-base	25	0.00816803	0.060621262	0.134738697	1
    # Chronos-tiny	25	0.006505346	0.712534046	0.009129874	1
    # Timer-UTSD	100	0.02019453	0.014750576	1.369067246	1
    # MOIRAI-base	100	0.021022844	0.07658062	0.274519119	1
    # Chronos-tiny	100	0.023592281	0.999940014	0.023593697	1
    # Timer-UTSD	500	0.079795599	0.054897022	1.453550588	1
    # MOIRAI-base	500	0.081245708	0.284106827	0.285968871	1
    # Chronos-tiny	500	0.085263395	3.700729895	0.023039616	1
    # Timer-UTSD	1	0.003139496	0.010003042	0.313854104	2
    # MOIRAI-base	1	0.001839399	0.052675533	0.034919425	2
    # Chronos-tiny	1	0.001800585	0.68881197	0.002614044	2
    # Timer-UTSD	25	0.008668518	0.011615658	0.746278705	2

    # Inference Overhead by Inference Batch Size and Model
    # 按照model和batch_size进行聚合
    df_duration = duration_df.groupby(['model_name', 'batch_size']).mean().reset_index()
    print(f'df_duration=\n{df_duration}')

    generate_latex_table(df_data_fix_params_mean, df_data_fix_samples_mean, df_duration, models)


if __name__ == "__main__":
    parent_dir_name = '/Users/cenzhiyao/PycharmProjects/ts_adaptive_inference/new_moti'

    # children_dir_names = get_dirs_from_range(
    #     parent_dir_name,
    #     '240711-135140-AnyTransform-hope-seed10!!!',
    #     '240716-004642-P500-S500-ABnone-SD4'
    # )

    # FIXME: MP=1
    children_dir_names = get_dirs_from_range(
        parent_dir_name,
        '240718-102827-P100-S100-ABnone-SD0-MP1',
        # '240718-145958-P500-S100-ABnone-SD4-MP1'
        '240718-163036-P500-S100-ABnone-SD9-MP1'
    )
    main(parent_dir_name, children_dir_names)
