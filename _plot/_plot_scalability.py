import atexit
import logging
import os
import signal
import sys
from datetime import datetime
from math import ceil

import pandas as pd
import pynvml
import numpy as np
from matplotlib import pyplot as plt

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
        'samples',
        'params',
        'ablation',
        'seed',
        'improve_percent_mean_mse',
        'improve_percent_mean_mae',
        'improve_percent_median_mse',
        'improve_percent_median_mae',
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
        improve_percent_mean_mse = df['improve_percent_mean_mse'].values[0]
        improve_percent_mean_mae = df['improve_percent_mean_mae'].values[0]
        improve_percent_median_mse = df['improve_percent_median_mse'].values[0]
        improve_percent_median_mae = df['improve_percent_median_mae'].values[0]
        df_data.loc[len(df_data)] = [num_samples, num_params, ablation, seed,
                                     improve_percent_mean_mse, improve_percent_mean_mae,
                                     improve_percent_median_mse, improve_percent_median_mae]

    df_data = df_data.drop(columns=['improve_percent_mean_mae', 'improve_percent_median_mae', 'seed'])
    df_data = df_data[df_data['ablation'] == 'none']

    # Define ranges for samples and params
    nums = [25, 50, 100, 250, 500]
    fix_num_sample = np.median(nums)
    fix_num_param = np.median(nums)

    # Group by number of samples (fixed params)
    df_data_fix_num_params = df_data[df_data['params'] == fix_num_param]
    print(f"df_data_fix_num_params: {df_data_fix_num_params}")
    df_data_mean_by_samples = df_data_fix_num_params.groupby('samples').mean().reset_index()
    df_data_std_by_samples = df_data_fix_num_params.groupby('samples').std().reset_index()

    # Group by number of params (fixed samples)
    df_data_fix_num_samples = df_data[df_data['samples'] == fix_num_sample]
    print(f"df_data_fix_num_samples: {df_data_fix_num_samples}")
    df_data_mean_by_params = df_data_fix_num_samples.groupby('params').mean().reset_index()
    df_data_std_by_params = df_data_fix_num_samples.groupby('params').std().reset_index()

    # Plot scalability with number of samples (Mean MSE)
    fig, ax = plt.subplots()
    x_samples = df_data_mean_by_samples['samples']
    mean_mse_samples = df_data_mean_by_samples['improve_percent_mean_mse']
    std_mse_samples = df_data_std_by_samples['improve_percent_mean_mse']

    ax.errorbar(x_samples, mean_mse_samples, yerr=std_mse_samples, fmt='-o', label='MSE_Mean %Imp',
                color='#2878B5', capsize=5)
    ax.set_title('Scalability with Number of Samples (Mean MSE)')
    ax.set_ylabel('Percentage Improvement')
    ax.set_xlabel('Number of Samples')
    ax.set_xticks(x_samples)
    ax.legend(loc='upper left', title=None)
    ax.set_ylim(0, 11)
    fig.tight_layout()
    plt.savefig('scalability_samples_mean_mse.png')
    plt.close(fig)

    # Plot scalability with number of samples (Median MSE)
    fig, ax = plt.subplots()
    median_mse_samples = df_data_mean_by_samples['improve_percent_median_mse']
    std_median_mse_samples = df_data_std_by_samples['improve_percent_median_mse']

    ax.errorbar(x_samples, median_mse_samples, yerr=std_median_mse_samples, fmt='-s', label='MSE_Median %Imp',
                color='#9AC9DB', capsize=5)
    ax.set_title('Scalability with Number of Samples (Median MSE)')
    ax.set_ylabel('Percentage Improvement')
    ax.set_xlabel('Number of Samples')
    ax.set_xticks(x_samples)
    ax.legend(loc='upper left', title=None)
    ax.set_ylim(0, 11)
    fig.tight_layout()
    plt.savefig('scalability_samples_median_mse.png')
    plt.close(fig)

    # Plot scalability with number of parameters (Mean MSE)
    fig, ax = plt.subplots()
    x_params = df_data_mean_by_params['params']
    mean_mse_params = df_data_mean_by_params['improve_percent_mean_mse']
    std_mse_params = df_data_std_by_params['improve_percent_mean_mse']

    ax.errorbar(x_params, mean_mse_params, yerr=std_mse_params, fmt='-o', label='MSE_Mean %Imp',
                color='#2878B5', capsize=5)
    ax.set_title('Scalability with Number of Parameters (Mean MSE)')
    ax.set_ylabel('Percentage Improvement')
    ax.set_xlabel('Number of Parameters')
    ax.set_xticks(x_params)
    ax.legend(loc='upper left', title=None)
    ax.set_ylim(0, 11)
    fig.tight_layout()
    plt.savefig('scalability_params_mean_mse.png')
    plt.close(fig)

    # Plot scalability with number of parameters (Median MSE)
    fig, ax = plt.subplots()
    median_mse_params = df_data_mean_by_params['improve_percent_median_mse']
    std_median_mse_params = df_data_std_by_params['improve_percent_median_mse']

    ax.errorbar(x_params, median_mse_params, yerr=std_median_mse_params, fmt='-s', label='MSE_Median %Imp',
                color='#9AC9DB', capsize=5)
    ax.set_title('Scalability with Number of Parameters (Median MSE)')
    ax.set_ylabel('Percentage Improvement')
    ax.set_xlabel('Number of Parameters')
    ax.set_xticks(x_params)
    ax.legend(loc='upper left', title=None)
    ax.set_ylim(0, 11)
    fig.tight_layout()
    plt.savefig('scalability_params_median_mse.png')
    plt.close(fig)


if __name__ == "__main__":
    parent_dir_name = '/Users/cenzhiyao/PycharmProjects/ts_adaptive_inference/new_moti'

    children_dir_names = get_dirs_from_range(parent_dir_name,
                                             '240711-135140-AnyTransform-hope-seed10!!!',
                                             '240712-200849-P25-S100-ABnone-SD2')
    main()
