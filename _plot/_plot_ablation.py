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
        'Trimmer',
        'Sampler',
        'Aligner',
        'Normalizer',
        'Warper',
        'Differentiator',
        'Inputer',
        'Denoiser',
        'Clipper',

        # 'Pipeline',

        # 'MultiMetric',
        # 'Stat',
        # 'Val',
        # 'Aug',
        # 'HPO',

    ]
    ablation_rename = {
        'none': 'All',
        'Trimmer': 'w/o Trimmer',
        'Sampler': 'w/o Sampler',
        'Aligner': 'w/o Aligner',
        'Normalizer': 'w/o Normalizer',
        'Warper': 'w/o Warper',
        'Differentiator': 'w/o Differencer',
        'Inputer': 'w/o Imputator',
        'Denoiser': 'w/o Denoiser',
        'Clipper': 'w/o Clipper',

        'Pipeline': 'w/o Reorder',
        'MultiMetric': 'w/o MultiMetric',
        'Stat': 'w/o Stat',
        'Val': 'w/o Val',
        'Aug': 'w/o Aug',
        'HPO': 'w/o HPO',
    }
    print(ablation_rename)
    import pandas as pd
    df_data = pd.DataFrame(columns=columns)

    for child_dir_name in children_dir_names:
        tokens = child_dir_name.split('-')
        # num_params = [int(token[1:]) for token in tokens if token.startswith('P')][0]
        # num_samples = [int(token[1:]) for token in tokens if token.startswith('S')][0]
        ablation = [token[2:] for token in tokens if token.startswith('AB')][0]
        seed = [int(token[2:]) for token in tokens if token.startswith('SD')][0]

        # FIXME：seed=1 随机性有点大。。。
        # if seed != 0:
        #     continue

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

    df_data = df_data.drop(columns=['improve_percent_mean_mae', 'improve_percent_median_mae','seed'])

    # 排一下ablations的顺序：
    df_data['ablation'] = pd.Categorical(df_data['ablation'], categories=ablations, ordered=True)
    # 把不同seed聚合在一起 (mean和std都要求)
    df_data_mean_by_ablation = df_data.groupby('ablation').mean().reset_index()
    print("df_data_mean_by_ablation")
    print(df_data_mean_by_ablation.to_string())

    df_data_std_by_ablation = df_data.groupby('ablation').std().reset_index()
    print("df_data_std_by_ablation")
    print(df_data_std_by_ablation.to_string())

    # # 绘制箱线图
    # import pandas as pd
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.boxplot(x='ablation', y='improve_percent_mean_mse', data=df_data_mean_by_ablation, showfliers=True)
    # plt.title('Ablation Effects on Improve Percentage Mean MSE')
    # plt.xlabel('Ablation Method')
    # plt.ylabel('Improve Percentage Mean MSE')
    # plt.show()

    # 画个柱状图
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(df_data_mean_by_ablation))
    width = 0.35

    fig = plt.figure(figsize=(8, 4))
    # fig = plt.figure(figsize=(10, 6))
    # fig, ax = plt.subplots()
    ax = fig.add_subplot(111)
    # Plot the first group of bars with full color
    ax.bar(x[0] - 1 / 2 * width, df_data_mean_by_ablation['improve_percent_mean_mse'][0], width,
           label='MSE Mean %Promotion', color='#2878B5')
    ax.bar(x[0] + 1 / 2 * width, df_data_mean_by_ablation['improve_percent_median_mse'][0], width,
           label='MSE Median %Promotion', color='#9AC9DB')

    # Plot the remaining groups of bars with lighter color (increased transparency)
    for i in range(1, len(x)):
        ax.bar(x[i] - 1 / 2 * width, df_data_mean_by_ablation['improve_percent_mean_mse'][i], width,
               label='_nolegend_', color='#2878B5', alpha=0.5)
        ax.bar(x[i] + 1 / 2 * width, df_data_mean_by_ablation['improve_percent_median_mse'][i], width,
               label='_nolegend_', color='#9AC9DB', alpha=0.5)

    # Add horizontal dashed lines at the values of the first group's bars
    ax.axhline(df_data_mean_by_ablation['improve_percent_mean_mse'][0], color='#2878B5', linestyle='--')
    ax.axhline(df_data_mean_by_ablation['improve_percent_median_mse'][0], color='#9AC9DB', linestyle='--')

    fontsize = 20
    fontsize2 = 15
    ax.set_title('Ablation Effects of Transform Operators', fontsize=fontsize)
    # ax.set_ylabel('Relative Percentage Promotion', fontsize=fontsize)
    ax.set_ylabel('%Promotion', fontsize=fontsize)
    ax.set_xticks(x)
    # 希望xticks的右端对齐刻度线 ...
    ax.set_xticklabels([ablation_rename[ablation] for ablation in df_data_mean_by_ablation['ablation']], fontsize=fontsize2, ha='right')
    plt.xticks(rotation=25)  # Tilt the x-axis labels by 45 degrees
    # y ticks 带百分比
    ax.set_yticklabels([f'{int(y)}%' for y in ax.get_yticks()], fontsize=fontsize2)
    ax.set_ylim(0, 14)  # Set the desired y-axis range
    # ax.set_ylim(0, 11)  # Set the desired y-axis range
    ax.legend(loc='upper left', title=None, fontsize=fontsize2)
    fig.tight_layout()
    # plt.show()
    # 保存pdf
    plt.savefig('ablation1.pdf')


if __name__ == "__main__":
    # 整合多次实验的结果数据，然后按照SEED聚合，最后封装成late格式
    # parent_dir_name = '/Users/cenzhiyao/Desktop/save/240612!!!-3'
    parent_dir_name = '/Users/cenzhiyao/PycharmProjects/ts_adaptive_inference/new_moti'

    children_dir_names = get_dirs_from_range(parent_dir_name,
                                             '240711-135140-P100-S100-ABnone-SD0',
                                             '240712-112525-P100-S100-ABPipeline-SD9')

    main()
