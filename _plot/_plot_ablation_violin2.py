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
        'Pipeline',
        'MultiMetric',
        'Stat',
        'Val',
        'Aug',
        'HPO',
    ]
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

    # Melt the DataFrame to long-form for Seaborn
    df_melted = df_data.melt(id_vars=['ablation', 'seed'],
                             value_vars=['improve_percent_mean_mse', 'improve_percent_median_mse'],
                             var_name='Metric', value_name='Improvement')

    # Define the categories
    categories = {
        'none': 'None',
        'Trimmer': 'Operator', 'Sampler': 'Operator', 'Aligner': 'Operator', 'Normalizer': 'Operator',
        'Warper': 'Operator', 'Differentiator': 'Operator', 'Inputer': 'Operator', 'Denoiser': 'Operator',
        'Clipper': 'Operator',
        'Pipeline': 'Framework', 'MultiMetric': 'Framework', 'Stat': 'Framework', 'Val': 'Framework',
        'Aug': 'Framework', 'HPO': 'Framework'
    }

    df_melted['Category'] = df_melted['ablation'].map(categories)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Define color palette for categories
    category_palette = {
        'None': '#2878B5',
        'Operator': '#9AC9DB',
        'Framework': '#FF800E'
    }

    metric_palette = {
        'improve_percent_mean_mse': '#FF800E',
        'improve_percent_median_mse': '#2878B5'
    }

    # sns.violinplot(ax=ax, x='ablation', y='Improvement', hue=['Metric','Category'], data=df_melted, inner=None, palette=[metric_palette,category_palette])
    # #
    # # Strip plot with category differentiation and transparency
    # sns.stripplot(ax=ax, x='ablation', y='Improvement', hue='Metric', data=df_melted, dodge=True, jitter=True,
    #               linewidth=1, alpha=0.7, palette='dark')


    # Violin plot with category differentiation
    sns.violinplot(ax=ax, x='ablation', y='Improvement', hue='Metric', data=df_melted, inner=None, palette='muted')
    #
    # Strip plot with category differentiation and transparency
    sns.stripplot(ax=ax, x='ablation', y='Improvement', hue='Metric', data=df_melted, dodge=True, jitter=True,
                  linewidth=1, alpha=0.7, palette='dark')


    # Violin plot with category differentiation
    # sns.violinplot(ax=ax, x='ablation', y='Improvement', hue='Category', data=df_melted, inner=None,
    #                palette='muted')

    # Strip plot with category differentiation
    # sns.stripplot(ax=ax, x='ablation', y='Improvement', hue='Category', data=df_melted, dodge=True, jitter=True,
    #               linewidth=1, palette='muted')


    ax.set_ylim(0, 15)  # Set the desired y-axis range

    ax.set_title('Improvement Percentage by Ablation')
    ax.set_xlabel('Ablation')
    ax.set_ylabel('Improvement Percentage')

    # Manually add category legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=category_palette['None'], label='None', marker='o', linestyle=''),
        Line2D([0], [0], color=category_palette['Operator'], label='Operator', marker='o', linestyle=''),
        Line2D([0], [0], color=category_palette['Framework'], label='Framework', marker='o', linestyle='')
    ]

    ax.legend(handles=legend_elements, loc='upper left', title='Category')

    plt.xticks(rotation=75)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 整合多次实验的结果数据，然后按照SEED聚合，最后封装成late格式
    # parent_dir_name = '/Users/cenzhiyao/Desktop/save/240612!!!-3'
    parent_dir_name = '/Users/cenzhiyao/PycharmProjects/ts_adaptive_inference/new_moti'

    children_dir_names = get_dirs_from_range(parent_dir_name,
                                             '240711-135140-P100-S100-ABnone-SD0',
                                             '240712-112525-P100-S100-ABPipeline-SD9')

    main()
