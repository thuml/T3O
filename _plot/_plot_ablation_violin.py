import atexit
import logging
import os
import signal
import sys
from datetime import datetime
from math import ceil

import pandas as pd
import pynvml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib


def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None


def cmp_time_str(str1, str2):
    time_format = "%y%m%d-%H%M%S"  # Define the time format based on the provided string
    time1 = datetime.strptime(str1, time_format)
    time2 = datetime.strptime(str2, time_format)
    return (time1 - time2).total_seconds()


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
    # 遍历所有子目录
    for child_dir in os.listdir(parent_dir):
        tokens = child_dir.split('-')
        if len(tokens) < 6:
            continue

        time_cur = trans_time_str(child_dir)
        if time_start <= time_cur <= time_end:
            child_dirs.append(child_dir)
    print('child_dirs', child_dirs)
    return child_dirs


def main():
    columns = [
        'ablation', 'seed',
        'improve_percent_mean_mse',
        'improve_percent_mean_mae',
        'improve_percent_median_mse',
        'improve_percent_median_mae',
    ]
    ablations = [
        'none',
        # 'Trimmer',
        # 'Sampler',
        # 'Aligner',
        # 'Normalizer',
        # 'Warper',
        # 'Differentiator',
        # 'Inputer',
        # 'Denoiser',
        # 'Clipper',
        #

        'MultiMetric',
        'Stat',
        'Aug',
        'Val',
        'Pipeline',  # FIXME 忘加了
        'HPO',
    ]
    ablation_rename = {
        'none': 'All',
        'Trimmer': 'w/o Trimmer',
        'Sampler': 'w/o Sampler',
        'Aligner': 'w/o Aligner',
        'Normalizer': 'w/o Normalizer',
        'Warper': 'w/o Warper',
        'Differentiator': 'w/o Differentiator',
        'Inputer': 'w/o Inputer',
        'Denoiser': 'w/o Denoiser',
        'Clipper': 'w/o Clipper',
        'Pipeline': 'w/o Reorder',
        'MultiMetric': 'w/o MultiMetric',
        'Stat': 'w/o MultiStatistics',
        'Val': 'w/o TwoStageRank',
        'Aug': 'w/o Augmentation',
        'HPO': 'w/o TPE',
    }

    import pandas as pd
    df_data = pd.DataFrame(columns=columns)

    for child_dir_name in children_dir_names:
        tokens = child_dir_name.split('-')
        # num_params = [int(token[1:]) for token in tokens if token.startswith('P')][0]
        # num_samples = [int(token[1:]) for token in tokens if token.startswith('S')][0]
        ablation = [token[2:] for token in tokens if token.startswith('AB')][0]
        seed = [int(token[2:]) for token in tokens if token.startswith('SD')][0]

        file_path = os.path.join(parent_dir_name, child_dir_name, '_exp_summary_total.csv')
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        improve_percent_mean_mse = df['improve_percent_mean_mse'].values[0]
        improve_percent_mean_mae = df['improve_percent_mean_mae'].values[0]
        improve_percent_median_mse = df['improve_percent_median_mse'].values[0]
        improve_percent_median_mae = df['improve_percent_median_mae'].values[0]
        df_data.loc[len(df_data)] = [ablation, seed, improve_percent_mean_mse, improve_percent_mean_mae,
                                     improve_percent_median_mse, improve_percent_median_mae]

    # 丢弃mae的列
    df_data = df_data.drop(columns=['improve_percent_mean_mae', 'improve_percent_median_mae'])

    # 排一下ablations的顺序：
    df_data['ablation'] = pd.Categorical(df_data['ablation'], categories=ablations, ordered=True)
    # 把不同seed聚合在一起 (mean和std都要求)
    df_data_mean_by_ablation = df_data.groupby('ablation').mean().reset_index()
    print("df_data_mean_by_ablation")
    print(df_data_mean_by_ablation.to_string())

    df_data_std_by_ablation = df_data.groupby('ablation').std().reset_index()
    print("df_data_std_by_ablation")
    print(df_data_std_by_ablation.to_string())

    # 绘制小提琴图和条形图
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    palette_dict = {
        'improve_percent_mean_mse': '#2878B5',
        'improve_percent_median_mse': '#9AC9DB',
    }

    # Melt the DataFrame to long-form for Seaborn
    df_melted = df_data.melt(id_vars=['ablation', 'seed'],
                             value_vars=['improve_percent_mean_mse', 'improve_percent_median_mse'],
                             var_name='Metric', value_name='Improvement')

    fig, ax = plt.subplots(figsize=(8, 4))
    # fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot
    sns.violinplot(ax=ax, x='ablation', y='Improvement', hue='Metric', data=df_melted, split=True, palette=palette_dict)

    # Strip plot
    sns.stripplot(ax=ax, x='ablation', y='Improvement', hue='Metric', data=df_melted, dodge=True, jitter=True,
                  linewidth=1, palette=palette_dict, alpha=[1])

    # Adjust transparency for groups except the first one
    count = 0
    for patch in ax.collections:
        if isinstance(patch, matplotlib.collections.PolyCollection):
            # Ignore the first group (x=0.0)
            # print(patch.get_paths()[0].vertices[0][0])
            # if patch.get_paths()[0].vertices[0][0] not in [0.011205302659472463, 0]:
            # if count > 2:
            #     patch.set_alpha(0.5)
            pass
        count += 1

    # Adjust transparency for strip plot points
    for line in ax.lines:
        if hasattr(line, 'get_xdata'):
            line.set_color('orange')
            if line.get_xdata()[0] != 0.0:  # Ignore the first group (x=0.0)
                line.set_alpha(0.5)
                # line.set_color('red')

    fontsize = 20
    fontsize2 = 15
    ax.set_title('Ablation Effects of Pipeline Steps', fontsize=fontsize)
    # ax.set_ylabel('Relative Percentage Promotion', fontsize=fontsize)
    ax.set_ylabel('%Promotion', fontsize=fontsize)
    ax.set_xlabel('')
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    labels_rename = {'improve_percent_mean_mse': 'MSE Mean %Promotion',
                     'improve_percent_median_mse': 'MSE Median %Promotion'}
    unique_labels = {labels_rename[label]: handle for label, handle in unique_labels.items()}
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', title=None, fontsize=fontsize2)
    plt.xticks(rotation=15)
    ax.set_xticklabels([ablation_rename[ablation] for ablation in df_data_mean_by_ablation['ablation']],
                       fontsize=fontsize2, ha='right')
    ax.set_yticklabels([f'{int(y)}%' for y in ax.get_yticks()], fontsize=fontsize2)
    ax.set_ylim(0, 18)  # Set the desired y-axis range
    # ax.set_ylim(0, 24)  # Set the desired y-axis range
    # ax.set_ylim(0, 16)  # Set the desired y-axis range
    # ax.set_yticklabels([f'{y}%' for y in ax.get_yticks()], fontsize=fontsize)
    fig.tight_layout()
    # plt.show()
    plt.savefig('ablation2.pdf')


if __name__ == "__main__":
    # 整合多次实验的结果数据，然后按照SEED聚合，最后封装成late格式
    # parent_dir_name = '/Users/cenzhiyao/Desktop/save/240612!!!-3'
    parent_dir_name = '/Users/cenzhiyao/PycharmProjects/ts_adaptive_inference/new_moti'

    children_dir_names = get_dirs_from_range(parent_dir_name,
                                             '240711-135140-P100-S100-ABnone-SD0',
                                             '240712-113437-P100-S100-ABMultiMetric-SD9')

    main()
