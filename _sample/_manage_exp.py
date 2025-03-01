import os
import shutil
import sys
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
import torch
from tqdm import tqdm

fast_mode = True if sys.platform == 'darwin' else False

import subprocess
import concurrent.futures
from utils import *


def run_process(command, output_file):
    """启动一个进程并执行给定的命令，同时将输出重定向到指定的文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        try:
            process = subprocess.Popen(command, stdout=file, stderr=file, shell=True)
            process.wait()
            return process.returncode
        except Exception as e:
            logging.error(f"An error occurred while running command '{command}': {e}")
            return -1


def main():
    t = time_start()
    # res_root_dir的子目录datetime如 ./new_moti/240515-204011
    date_time_str = datetime.now().strftime("%y%m%d-%H%M%S")
    res_root_dir = os.path.join('new_moti' if not fast_mode else 'debug', date_time_str)
    # if os.path.exists(f'{res_root_dir}'):
    #     logging.warning(f"Directory '{res_root_dir}' already exists. Removing it.")
    #     shutil.rmtree(f'{res_root_dir}')
    os.makedirs(f'{res_root_dir}', exist_ok=True)
    # 设置日志记录
    log_file = os.path.join(res_root_dir, '_manage_exp.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler(log_file, 'w', 'utf-8'), logging.StreamHandler(sys.stdout)])

    gpu_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
    use_gpu = True if torch.cuda.is_available() else False
    # 其实主要限速步骤在cpu，还有些是在plot的时候
    # Ok: 很多时候running的会多于max_processes数量？但实际没有Running
    # Ok: max_processes大了之后(8) Timer预测很容易卡死！！！！！-> 可能是因为pred_len*96的bug
    max_processes = 24 if not fast_mode else 2  # FIXME:
    # min: 138->104->90->86
    # proc: 8->16->24->32
    # 但是32会导致的问题是第一次结果出现在20min左右 还可能被kill ->目前24最好
    # 单个运行好像17s？

    # 'Timer-LOTSA', 'Timer-UTSD'
    # 'Uni2ts-base' seq长度192以上会内存不够 -> fix 96 好像就行了
    # 'Unit2ts-large' 太消耗时间
    # 'Chronos-tiny' 还是慢 而且有各种问题..... # 可能是nan
    model_names = ['Timer-UTSD', 'Timer-LOTSA', 'Uni2ts-small', 'Uni2ts-base', 'Uni2ts-large'] \
        if not fast_mode else ['Uni2ts-small']  # FIXME:
    data_names = ['ETTm1', 'ETTh1', 'Exchange', 'Weather', 'Electricity', 'Traffic', 'ETTm2', 'ETTh2'] \
        if not fast_mode else ['ETTm1']  # FIXME:
    pred_lens = [96, 192, 336, 720] if not fast_mode else [96]  # FIXME:

    # 创建 DataFrame 用于记录实验状态和结果
    cmp_methods = ["org", "our", 'improve_percent']
    metric_names = ["mae", "mse", "rmse", "mape", "mspe"]
    metric_columns = [f"{cmp_method}_{metric_name}" for metric_name in metric_names for cmp_method in cmp_methods]
    basic_columns = ["data_name", "model_name", "pred_len"]
    status_columns = ["Command", "Result Directory", "Status", "Start Time", "End Time", "Duration"]
    experiment_columns = basic_columns + metric_columns + status_columns
    experiment_df = pd.DataFrame(columns=experiment_columns)
    # 初始化 DataFrame
    import itertools
    gpu_index_iter = itertools.cycle(gpu_indexes)
    for model_name, data_name, pred_len in itertools.product(model_names, data_names, pred_lens):
        gpu_index = next(gpu_index_iter) if use_gpu else None
        command = f" {f'CUDA_VISIBLE_DEVICES={gpu_index}' if use_gpu else ''} " \
                  f" python3 ./AnyTransform/single_exp.py --res_root_dir {res_root_dir}" \
                  f" {'--use_gpu' if use_gpu else ''} --gpu_indexes {gpu_index}" \
                  f" --data_name {data_name} --model_name {model_name} --pred_len {pred_len}" \
                  f" {'--fast_mode' if fast_mode else ''}"
        res_dir = os.path.join(res_root_dir, f"{data_name}", f"{model_name}", f"pred_len-{pred_len}")
        basic_values = [data_name, model_name, pred_len]
        metric_values = [None] * len(metric_columns)
        status_values = [command, res_dir, "Pending", None, None, None]
        experiment_df.loc[len(experiment_df)] = basic_values + metric_values + status_values

    # 创建进程池执行器
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        # 提交每个进程的命令到进程池
        # commands = experiment_df["Command"].values
        res_dirs = experiment_df["Result Directory"].values
        res_log_files = [os.path.join(res_dir, 'exp.log') for res_dir in res_dirs]
        futures = []
        # for command, res_log_file in zip(commands, res_log_files):
        #     future = executor.submit(run_process, command, res_log_file)
        #     futures.append(future)
        #     # 休息0.1s，避免max_processes和实际运行processes数量不一致 -》 好像没用
        #     logging.info(f"Command '{command}' submitted.")
        #     time.sleep(0.1)

        # 持续检查任务的状态和结果，直到所有任务完成
        initial_slow_interval = 60
        max_slow_interval = 300
        step_slow_interval = 60
        current_fast_interval = 10
        current_slow_interval = initial_slow_interval
        while True:
            logging.info("\nChecking experiment status...")
            save_and_display = False  # 本轮实验结束后，是否显示实验状态
            break_flag = False

            for exp_idx in range(len(futures)):
                future = futures[exp_idx]

                if future.running():
                    # 当对应任务的 Start Time 为空时，表示任务刚刚开始，填充实验开始时间和状态
                    if experiment_df.loc[exp_idx, "Start Time"] is None:
                        # 记录实验开始时间和状态
                        # start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # experiment_df.loc[exp_idx, "Status"] = "Running"
                        # experiment_df.loc[exp_idx, "Start Time"] = start_time_str
                        # logging.info(f"Experiment {exp_idx} started: {experiment_df.loc[exp_idx, 'Command']}")
                        raise ValueError("Running")
                    elif experiment_df.loc[exp_idx, "End Time"] is None:  # 对于开始了但是没结束的填充Duration。。。现在没用了
                        start_time_str = experiment_df.loc[exp_idx, "Start Time"]
                        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                        duration = datetime.now() - start_time
                        duration_minutes = duration.total_seconds() / 60
                        experiment_df.loc[exp_idx, "Duration"] = duration_minutes
                if future.done():
                    # 当对应任务的 End Time 为空时，表示任务刚刚完成，填充实验结束时间、状态和结果
                    if experiment_df.loc[exp_idx, "End Time"] is None:
                        end_time = datetime.now()
                        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
                        start_time_str = experiment_df.loc[exp_idx, "Start Time"]
                        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                        duration = end_time - start_time
                        duration_minutes = duration.total_seconds() / 60
                        returncode = future.result()
                        if returncode != 0:
                            cmd = experiment_df.loc[exp_idx, "Command"]
                            logging.error(f"Error occurred while running command '{cmd}'")
                            logging.error(f"Check log file '{res_log_files[exp_idx]}' for details.")
                            raise ValueError("Failed")
                        # 记录实验结束时间、状态和结果
                        experiment_df.loc[exp_idx, "Status"] = "Completed" if returncode == 0 else "Failed"
                        experiment_df.loc[exp_idx, "End Time"] = end_time_str
                        experiment_df.loc[exp_idx, "Duration"] = duration_minutes
                        logging.info(f"Experiment {exp_idx} completed: {experiment_df.loc[exp_idx, 'Command']}")

                        # 拼装得到的res_dir结果
                        data_name, model_name, pred_len = experiment_df.loc[exp_idx, basic_columns].values
                        res_dir = os.path.join(res_root_dir, data_name, model_name, f'pred_len-{pred_len}')
                        logging.info(f"res_dir={res_dir}")
                        summary_results = pd.read_csv(os.path.join(res_dir, 'summary_results.csv'))
                        # 将summary_results中的结果依据basic_values填充其metric_values到experiment_df中,依据basic_columns
                        # summary_results的列：basic_columns + metric_columns
                        # experiment_df的列：basic_columns + metric_columns + status_columns
                        assert len(summary_results) == 1, f"len(summary_results)={len(summary_results)} != 1"
                        metric_values = summary_results.loc[0, metric_columns].values
                        # logging.info(f"metric_values={metric_values}")
                        # logging.info(f"exp_idx={exp_idx}")
                        experiment_df.loc[exp_idx, metric_columns] = metric_values
                        save_and_display = True

            # 检查是否所有任务已完成
            statuses = experiment_df["Status"].values
            if "Running" not in statuses and "Pending" not in statuses:
                logging.info("\nAll experiments completed.")
                save_and_display = True
                break_flag = True

            # 检查当前Running的任务的个数，不足则继续提交任务
            waiting_task_num = np.sum(statuses == "Pending")
            non_waiting_task_num = np.sum(statuses != "Pending")
            running_task_num = np.sum(statuses == "Running")
            should_submit_num = min(max_processes - running_task_num, waiting_task_num)
            submit_num = 0
            while submit_num < should_submit_num:
                exp_idx = submit_num + non_waiting_task_num
                command, res_log_file = experiment_df.loc[exp_idx, "Command"], res_log_files[exp_idx]
                future = executor.submit(run_process, command, res_log_file)
                futures.append(future)
                start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                experiment_df.loc[exp_idx, "Status"] = "Running"
                experiment_df.loc[exp_idx, "Start Time"] = start_time_str
                logging.info(f"Experiment {exp_idx} started: {experiment_df.loc[exp_idx, 'Command']}")
                # logging.info(f"Command '{command}' submitted.")
                submit_num += 1
                save_and_display = True

            if save_and_display:
                # 保存实验状态到 CSV 文件
                experiment_df.to_csv(os.path.join(res_root_dir, "_experiment_status.csv"), index=False)
                # 设置 pandas 显示选项，不限制行和列的展示数量
                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)
                pd.options.display.float_format = '{:.2f}'.format  # 百分比看小数点后2位够了
                # 输出当前实验状态
                logging.info("\nExperiment Status:")
                display_columns = ["Result Directory", "Status", "Start Time", "Duration"]
                logging.info(experiment_df[display_columns][experiment_df["Status"] != "Pending"].to_string())
                # 展示实验总数和各状态的数量
                logging.info(f"Total experiments: {len(experiment_df)}")
                for status in ["Pending", "Running", "Completed", "Failed"]:
                    status_num = np.sum(experiment_df["Status"] == status)
                    logging.info(f"{status} experiments: {status_num}")
                logging.info("\nExperiment Metrics:")

                # 构造exp_summary（计算每个 perspective (model_name / data_name / pred_len) 的 mse和mae的improvement_rate的均值）
                # 注意：perspective_type = 'model_name' / 'data_name' / 'pred_len'
                completed_exps = experiment_df[experiment_df['Status'] == 'Completed']
                if len(completed_exps) != 0:
                    columns_short_dict = {
                        "improve_percent_mse": "Imp_MSE",
                        "improve_percent_mae": "Imp_MAE",
                        "better_percent_mse": "Bet_MSE",
                        "better_percent_mae": "Bet_MAE"
                    }
                    # experiment_df_short = experiment_df.rename(columns=display_columns_dict)
                    short_display_columns = ["Res Dir", "Status", "Start", "Dur"]
                    perspectives = ['model_name', 'data_name', 'pred_len']
                    agg_dict = {
                        'improve_percent_mse': 'mean',
                        'improve_percent_mae': 'mean',
                        'Duration': 'mean',
                    }
                    for perspective in perspectives:
                        summary_df = completed_exps.groupby(perspective).agg(agg_dict).reset_index()
                        # 计算improvement_percent_mse大于0的百分比
                        summary_df['better_percent_mse'] = completed_exps.groupby(perspective)['improve_percent_mse'] \
                            .apply(lambda x: (x >= 0).mean() * 100).values
                        # 计算improvement_percent_mae大于0的百分比
                        summary_df['better_percent_mae'] = completed_exps.groupby(perspective)['improve_percent_mae'] \
                            .apply(lambda x: (x >= 0).mean() * 100).values
                        summary_df_path = os.path.join(res_root_dir, f"_exp_summary_{perspective}.csv")
                        summary_df.to_csv(summary_df_path, index=False)
                        summary_df_short = summary_df.rename(columns=columns_short_dict)
                        logging.info(summary_df_short.to_string())
                    # # total_summary 聚合成一条数据
                    # total_summary_df = completed_exps[list(agg_dict.keys())].agg(agg_dict).reset_index()
                    # # 计算improvement_percent_mse大于0的百分比
                    # total_summary_df['better_percent_mse'] = (completed_exps['improve_percent_mse'] >= 0).mean() * 100
                    # # 计算improvement_percent_mae大于0的百分比
                    # total_summary_df['better_percent_mae'] = (completed_exps['improve_percent_mae'] >= 0).mean() * 100
                    # 聚合数据
                    total_summary = completed_exps[list(agg_dict.keys())].agg(agg_dict)
                    # 计算 improvement_percent_mse 和 improvement_percent_mae 大于 0 的百分比
                    better_percent_mse = (completed_exps['improve_percent_mse'] > 0).mean() * 100
                    better_percent_mae = (completed_exps['improve_percent_mae'] > 0).mean() * 100
                    # 将结果转为 DataFrame，并添加百分比列
                    better_percent_series = pd.Series({'better_percent_mse': better_percent_mse,
                                                       'better_percent_mae': better_percent_mae})
                    total_summary_df = pd.concat([total_summary, better_percent_series], axis=0) \
                        .to_frame(name='Value').reset_index()
                    # 保存结果到 CSV 文件
                    total_summary_df_path = os.path.join(res_root_dir, "_exp_summary_total.csv")
                    total_summary_df.to_csv(total_summary_df_path, index=False)
                    total_summary_df_short = total_summary_df.rename(columns=columns_short_dict)
                    logging.info(total_summary_df_short.to_string())

            # log_time_delta(t, "main")
            exp_duration = time.time() - t
            logging.info("Experiment duration: {:.2f} minutes".format(exp_duration / 60))

            if break_flag:
                break

            # 间隔一段时间后再检查任务状态
            if save_and_display:
                current_slow_interval = initial_slow_interval
                logging.info(f"Waiting for {current_fast_interval} seconds...")
                time.sleep(current_fast_interval)
            else:
                logging.info("No change in experiment status.")
                logging.info(f"Waiting for {current_slow_interval} seconds...")
                time.sleep(current_slow_interval)
                current_slow_interval = min(current_slow_interval + step_slow_interval, max_slow_interval)


if __name__ == "__main__":
    main()
