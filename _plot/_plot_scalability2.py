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

    print('child_dirs', child_dirs)
    return child_dirs


# def plot_scalability(df, x_col, y_col_mean, y_col_median, xlabel, ylabel, title, filename, color_mean, color_median):
def plot_scalability(df_y, df_std, x_col, y_col_mean, y_col_median, xlabel, ylabel, title, filename, color_mean,
                     color_median):
    fontsize = 20
    fontsize2 = 15
    # fontsize = 18
    # fontsize2 = 14
    # 限制fig大小
    fig = plt.figure(figsize=(8, 4))
    # fig = plt.figure(figsize=(4, 4))
    # fig = plt.figure(figsize=(5, 4))
    # fig = plt.figure(figsize=(8, 4))
    # fig = plt.figure(figsize=(10, 4))
    # fig, ax = plt.subplots()
    ax = fig.add_subplot(111)
    x = df_y[x_col]
    y_mean = df_y[y_col_mean]
    y_median = df_y[y_col_median]
    std_mean = df_std[y_col_mean]
    std_median = df_std[y_col_median]

    ax.plot(x, y_mean, '-o', label=f'{ylabel} Mean %Promotion', color=color_mean)
    ax.plot(x, y_median, '-s', label=f'{ylabel} Median %Promotion', color=color_median)

    # # 重叠，太丑了（方差并没有逐渐变小，都差不多。。。
    ax.fill_between(x, y_mean - std_mean, y_mean + std_mean, color=color_mean, alpha=0.1)
    ax.fill_between(x, y_median - std_median, y_median + std_median, color=color_median, alpha=0.1)

    # ax.set_xscale('log')
    ax.set_title(title, fontsize=fontsize)
    # AttributeError: 'Text' object has no property 'font_size'
    ax.set_ylabel('%Promotion', fontsize=fontsize)
    ax.set_xlabel(f'{xlabel}', fontsize=fontsize)
    visible_ticks = [1, 50, 100, 175, 250, 375, 500]
    ax.set_xticks(visible_ticks, minor=False)
    ax.set_xticklabels(visible_ticks, fontsize=fontsize2)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.legend(loc='upper left', fontsize=fontsize2)
    # 右下角
    ax.legend(loc='lower right', fontsize=fontsize2)
    ax.set_ylim(0, 12)
    # 设置ytick的字体大小
    ax.set_yticklabels([f'{y}%' for y in ax.get_yticks()], fontsize=fontsize2)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def main(parent_dir_name, children_dir_names):
    columns = [
        'samples', 'params', 'ablation', 'seed', 'improve_percent_mean_mse',
        'improve_percent_mean_mae', 'improve_percent_median_mse', 'improve_percent_median_mae'
    ]

    df_data = pd.DataFrame(columns=columns)

    for child_dir_name in children_dir_names:
        tokens = child_dir_name.split('-')
        ablation = [token[2:] for token in tokens if token.startswith('AB')][0]
        num_params = [int(token[1:]) for token in tokens if token.startswith('P')][0]
        num_samples = [int(token[1:]) for token in tokens if token.startswith('S') and not token.startswith('SD')][0]
        seed = [int(token[2:]) for token in tokens if token.startswith('SD')][0]

        file_path = os.path.join(parent_dir_name, child_dir_name, '_exp_summary_total.csv')
        df = pd.read_csv(file_path)

        df_data.loc[len(df_data)] = [
            num_samples, num_params, ablation, seed,
            df['improve_percent_mean_mse'].values[0],
            df['improve_percent_mean_mae'].values[0],
            df['improve_percent_median_mse'].values[0],
            df['improve_percent_median_mae'].values[0]
        ]

    df_data = df_data.drop(columns=['improve_percent_mean_mae', 'improve_percent_median_mae', 'seed'])
    df_data = df_data[df_data['ablation'] == 'none']

    # nums = [25, 50, 100, 250, 500]
    # visible_num = [10, 25, 37, 50, 75, 100, 175, 250, 375, 500]
    # df_data = df_data[df_data['samples'].isin(visible_num) & df_data['params'].isin(visible_num)]
    fix_num_sample = 100
    fix_num_param = 100

    df_data_fix_num_params = df_data[df_data['params'] == fix_num_param]
    df_data_mean_by_samples = df_data_fix_num_params.groupby('samples').mean().reset_index()
    print(f'df_data_mean_by_samples=\n{df_data_mean_by_samples}')
    df_data_std_by_samples = df_data_fix_num_params.groupby('samples').std().reset_index()
    print(f'df_data_std_by_samples=\n{df_data_std_by_samples}')

    df_data_fix_num_samples = df_data[df_data['samples'] == fix_num_sample]
    df_data_mean_by_params = df_data_fix_num_samples.groupby('params').mean().reset_index()
    print(f'df_data_mean_by_params=\n{df_data_mean_by_params}')
    df_data_std_by_params = df_data_fix_num_samples.groupby('params').std().reset_index()
    print(f'df_data_std_by_params=\n{df_data_std_by_params}')

    plot_scalability(
        df_data_mean_by_samples, df_data_std_by_samples, 'samples', 'improve_percent_mean_mse',
        'improve_percent_median_mse',
        'Data Samples', 'MSE', 'Scalability',
        'scalability_samples.pdf', '#2878B5', '#9AC9DB'
    )

    plot_scalability(
        df_data_mean_by_params, df_data_std_by_params, 'params', 'improve_percent_mean_mse',
        'improve_percent_median_mse',
        'Transform Trials', 'MSE', 'Scalability',
        'scalability_trials.pdf', '#2878B5', '#9AC9DB'
    )


if __name__ == "__main__":
    parent_dir_name = '/Users/cenzhiyao/PycharmProjects/ts_adaptive_inference/new_moti'

    children_dir_names = get_dirs_from_range(
        parent_dir_name,
        '240711-135140-AnyTransform-hope-seed10!!!',
        '240714-124640-P100-S10-ABnone-SD9'
    )
    main(parent_dir_name, children_dir_names)
