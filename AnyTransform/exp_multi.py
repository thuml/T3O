import atexit
import logging
import os
import signal
import sys
from datetime import datetime
from math import ceil
import argparse
import pandas as pd
import pynvml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from AnyTransform.utils import *

fast_mode = True if sys.platform == 'darwin' else False
logging.info(f"fast_mode={fast_mode}")

# import subprocess
# import concurrent.futures
from AnyTransform.executor import *

pynvml.nvmlInit()


def get_least_loaded_gpu(idx_list):
    device_count = pynvml.nvmlDeviceGetCount()
    min_load = float('inf')
    selected_gpu_idx = 0
    for idx in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        load = utilization.gpu + utilization.memory
        if load < min_load and idx in idx_list:
            min_load = load
            selected_gpu_idx = idx
    return selected_gpu_idx


def select_gpu_idx(idx_list, gpu_assigned_dict, max_processes):
    available_idx_list = []
    max_processes_per_gpu = ceil(max_processes / len(idx_list))
    for idx in idx_list:
        if gpu_assigned_dict[idx] < max_processes_per_gpu:
            available_idx_list.append(idx)
    idx = get_least_loaded_gpu(available_idx_list)
    logging.info(f"Scheduling experiment on GPU {idx}.")
    logging.debug(f"idx={idx}, idx_list={idx_list}, gpu_assigned_dict={gpu_assigned_dict}, "
                  f"max_processes={max_processes}, max_processes_per_gpu={max_processes_per_gpu}, "
                  f"available_idx_list={available_idx_list}")
    return idx


def main(model_names, data_names, pred_lens,
         num_params, num_samples, ablation, max_processes, seed):
    t = time_start()
    # res_root_dir的子目录datetime如 ./new_moti/240515-204011
    date_time_str = datetime.now().strftime("%y%m%d-%H%M%S")
    setting_suffix = f"-P{num_params}-S{num_samples}-AB{ablation}-SD{seed}-MP{max_processes}"
    res_root_dir = os.path.join('new_moti' if not fast_mode else 'debug', date_time_str + setting_suffix)
    # logging.info(f"res_root_dir={res_root_dir}")
    # if os.path.exists(f'{res_root_dir}'):
    #     logging.warning(f"Directory '{res_root_dir}' already exists. Removing it.")
    #     shutil.rmtree(f'{res_root_dir}')
    os.makedirs(f'{res_root_dir}', exist_ok=True)
    # FIXME: 设置日志记录 之后才能正常用logging！
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    log_file = os.path.join(res_root_dir, '_manage_exp.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler(log_file, 'w', 'utf-8'), logging.StreamHandler(sys.stdout)])
    logging.info(f"res_root_dir={res_root_dir}")

    gpu_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
    gpu_assigned_dict = {k: 0 for k in gpu_indexes}
    use_gpu = True if torch.cuda.is_available() else False
    # 其实主要限速步骤在cpu，还有些是在plot的时候
    # Ok: 很多时候running的会多于max_processes数量？但实际没有Running
    # Ok: max_processes大了之后(8) Timer预测很容易卡死！！！！！-> 可能是因为pred_len*96的bug
    # max_processes = 8 if not fast_mode else 2  # FIXME:
    # min: 138->104->90->86
    # proc: 8->16->24->32
    # 但是32会导致的问题是第一次结果出现在20min左右 还可能被kill ->目前24最好
    # 单个运行好像17s？
    # 1->0.67min/个
    # 8->3.85min/个
    # 高于8内存有限可能影响Chronos效果

    # 创建 DataFrame 用于记录实验状态和结果
    cmp_methods = ["org", "our",
                   'improve_percent1', 'improve_percent2',
                   'improve_percent_median', 'improve_percent_iqr',
                   'improve_percent_mean', 'improve_percent_std',
                   'improve_percent_max', 'improve_percent_min',
                   'better_percent', 'draw_percent',
                   'improve_percent_in_better', 'improve_percent_in_worse',
                   'improve_percent_in_hard', 'improve_percent_in_medium', 'improve_percent_in_easy']
    metric_names = ["mae", "mse", "rmse", "mape", "mspe"]
    metric_columns = [f"{cmp_method}_{metric_name}" for metric_name in metric_names for cmp_method in cmp_methods]
    basic_columns = ["data_name", "model_name", "pred_len"]
    status_columns = ["Command", "Result Directory", "Status", "Start Time", "End Time", "Duration"]
    params_columns = ['num_params'] + list(get_params_space_and_org(fast_mode)[0].keys())
    duration_columns = ['adapt_duration', 'test_duration', 'adapt_duration_percent']
    experiment_columns = basic_columns + metric_columns + status_columns + params_columns + duration_columns
    experiment_df = pd.DataFrame(columns=experiment_columns)
    # 初始化 DataFrame
    import itertools
    # gpu_index_iter = itertools.cycle(gpu_indexes)
    for model_name, data_name, pred_len in itertools.product(model_names, data_names, pred_lens):
        # gpu_index = next(gpu_index_iter) if use_gpu else None
        # gpu_index = select_gpu_idx(gpu_indexes, set(experiment_df["gpu_indexes"].values))
        # command = f" {f'CUDA_VISIBLE_DEVICES={gpu_index}' if use_gpu else ''} " \
        #           f" python3 ./AnyTransform/exp_single.py --res_root_dir {res_root_dir}" \
        #           f" {'--use_gpu' if use_gpu else ''} --gpu_indexes {gpu_index}" \
        #           f" --data_name {data_name} --model_name {model_name} --pred_len {pred_len}" \
        #           f" {'--fast_mode' if fast_mode else ''}"
        res_dir = os.path.join(res_root_dir, f"{data_name}", f"{model_name}", f"pred_len-{pred_len}")
        basic_values = [data_name, model_name, pred_len]
        metric_values = [None] * len(metric_columns)
        status_values = [None, res_dir, "Pending", None, None, None]
        params_values = [None] * len(params_columns)
        duration_values = [None] * len(duration_columns)
        exp_values = basic_values + metric_values + status_values + params_values + duration_values
        experiment_df.loc[len(experiment_df)] = exp_values

    # 创建进程池执行器
    # 提交每个进程的命令到进程池
    # commands = experiment_df["Command"].values
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
    with MyExecutor() as my_executer:
        res_dirs = experiment_df["Result Directory"].values
        res_log_files = [os.path.join(res_dir, 'exp.log') for res_dir in res_dirs]
        futures = []
        atexit.register(atexit_handler, futures)
        signal.signal(signal.SIGCHLD, signal_handler)

        # 持续检查任务的状态和结果，直到所有任务完成
        # if max_processes <= 4:

        if True:
            initial_slow_interval = 10
            max_slow_interval = 300
            step_slow_interval = 5
            current_fast_interval = 10
        else:
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
                        command = experiment_df.loc[exp_idx, "Command"]
                        if returncode != 0:
                            logging.error(
                                f"Error occurred while running command '{command}', return code {returncode}.")
                            logging.error(f"Check log file '{res_log_files[exp_idx]}' for details.")
                            raise ValueError("Failed")
                        # 记录实验结束时间、状态和结果
                        experiment_df.loc[exp_idx, "Status"] = "Completed" if returncode == 0 else "Failed"
                        experiment_df.loc[exp_idx, "End Time"] = end_time_str
                        experiment_df.loc[exp_idx, "Duration"] = duration_minutes
                        logging.info(f"Experiment {exp_idx} completed: {command}")

                        # 拼装得到的res_dir结果
                        data_name, model_name, pred_len = experiment_df.loc[exp_idx, basic_columns].values
                        res_dir = os.path.join(res_root_dir, data_name, model_name, f'pred_len-{pred_len}')
                        logging.info(f"res_dir={res_dir}")
                        detailed_results = pd.read_csv(os.path.join(res_dir, 'detailed_results.csv'))
                        # 将summary_results中的结果依据basic_values填充其metric_values到experiment_df中,依据basic_columns
                        # summary_results的列：basic_columns + metric_columns
                        # experiment_df的列：basic_columns + metric_columns + status_columns
                        # FIXME: 只考虑OT的target_column的行的数值
                        target_results = detailed_results[detailed_results['target_column'] == 'OT']
                        assert len(target_results) == 1, f"len(target_results)={len(target_results)}"
                        # 填充metric_values和params_values
                        experiment_df.loc[exp_idx, metric_columns + params_columns + duration_columns] = \
                            target_results[metric_columns + params_columns + duration_columns].values[0]
                        save_and_display = True
                        if use_gpu:
                            gpu_index = int(command.split(' ')[0].split('=')[1])  # FIXME 认定只有一个
                            gpu_assigned_dict[gpu_index] -= 1  # 释放GPU
                            logging.info(f"gpu_assigned_dict[{gpu_index}] += 1, gpu_assigned_dict={gpu_assigned_dict}")

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
                res_log_file = res_log_files[exp_idx]
                data_name, model_name, pred_len = experiment_df.loc[exp_idx, basic_columns].values

                gpu_index = select_gpu_idx(gpu_indexes, gpu_assigned_dict, max_processes)
                gpu_assigned_dict[gpu_index] += 1
                logging.info(f"gpu_assigned_dict[{gpu_index}] += 1, gpu_assigned_dict={gpu_assigned_dict}")
                command = f"{f'CUDA_VISIBLE_DEVICES={gpu_index}' if use_gpu else ''} " \
                          f" python3 ./AnyTransform/exp_single.py --res_root_dir {res_root_dir}" \
                          f" {'--use_gpu' if use_gpu else ''} --gpu_indexes {gpu_index}" \
                          f" --data_name {data_name} --model_name {model_name} --pred_len {pred_len}" \
                          f" {'--fast_mode' if fast_mode else ''}" \
                          f" --num_params {num_params} --num_samples {num_samples} --ablation {ablation}" \
                          f" --seed {seed}"

                experiment_df.loc[exp_idx, "Command"] = command

                # future = executor.submit(run_process, command, res_log_file)
                future = my_executer.submit(run_command, command, res_log_file)
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
                logging.info(f"\n{experiment_df[display_columns][experiment_df['Status'] != 'Pending'].to_string()}")
                # 展示实验设置
                logging.info("\nExperiment Settings:")
                logging.info(f"num_params={num_params}, num_samples={num_samples}, ablation={ablation}, seed={seed}")
                # 展示实验总数和各状态的数量
                logging.info(f"Total experiments: {len(experiment_df)}")
                for status in ["Pending", "Running", "Completed", "Failed"]:
                    status_num = np.sum(experiment_df["Status"] == status)
                    logging.info(f"{status} experiments: {status_num}")
                logging.info(f"max_processes={max_processes}")
                logging.info("\nExperiment Metrics:")

                # 构造exp_summary（计算每个 perspective (model_name / data_name / pred_len) 的 mse和mae的improvement_rate的均值）
                # 注意：perspective_type = 'model_name' / 'data_name' / 'pred_len'
                completed_exps = experiment_df[experiment_df['Status'] == 'Completed']
                if len(completed_exps) != 0:
                    columns_short_dict = {
                        "model_name": "Model",
                        "data_name": "Data",
                        "pred_len": "PredLen",

                        # "improve_percent1_mse": "Imp%_MSE",
                        # "improve_percent2_mse": "Imp(SB)%_MSE",
                        "improve_percent_mean_mse": "Imp%_MSE_Mean",
                        "improve_percent_median_mse": "Imp%_MSE_Med",
                        "improve_percent_std_mse": "Imp%_MSE_Std",
                        "improve_percent_iqr_mse": "Imp%_MSE_IQR",
                        # "better_percent_mse": "Bett%_MSE",
                        # "draw_percent_mse": "Draw%_MSE",
                        # "improve_percent_in_better_mse": "Imp%_MSE(Bett)",
                        # "improve_percent_in_worse_mse": "Imp%_MSE(Wrse)",
                        # "improve_percent_in_hard_mse": "Imp%_MSE(InHard)",
                        # "improve_percent_in_medium_mse": "Imp%_MSE(Med)",
                        # "improve_percent_in_easy_mse": "Imp%_MSE(Easy)",

                        # "improve_percent1_mae": "Imp%_MAE",
                        # "improve_percent2_mae": "Imp(SB)%_MAE",
                        "improve_percent_mean_mae": "Imp%_MAE_Mean",
                        "improve_percent_median_mae": "Imp%_MAE_Med",
                        "improve_percent_std_mae": "Imp%_MAE_Std",
                        "improve_percent_iqr_mae": "Imp%_MAE_IQR",
                        # "better_percent_mae": "Bett%_MAE",
                        # "draw_percent_mae": "Draw%_MAE",
                        # "improve_percent_in_better_mae": "Imp%_MAE(Bett)",
                        # "improve_percent_in_worse_mae": "Imp%_MAE(Wrse)",
                        # "improve_percent_in_hard_mae": "Imp%_MAE(InHard)",
                        # "improve_percent_in_medium_mae": "Imp%_MAE(Med)",
                        # "improve_percent_in_easy_mae": "Imp%_MAE(Easy)",

                        "adapt_duration": "AdaptSec",
                        "num_params": "Params",
                        "Duration": "TotalMin",
                    }
                    agg_dict = {
                        # 'improve_percent1_mse': 'mean',
                        # 'improve_percent2_mse': 'mean',
                        # 'better_percent_mse': 'mean',
                        # 'draw_percent_mse': 'mean',
                        # 'improve_percent_in_better_mse': 'mean',
                        # 'improve_percent_in_worse_mse': 'mean',
                        # 'improve_percent_in_hard_mse': 'mean',
                        # 'improve_percent_in_medium_mse': 'mean',
                        # 'improve_percent_in_easy_mse': 'mean',
                        'improve_percent_mean_mse': 'mean',
                        'improve_percent_median_mse': 'mean',
                        'improve_percent_std_mse': 'mean',
                        'improve_percent_iqr_mse': 'mean',
                        'improve_percent_max_mse': 'mean',
                        'improve_percent_min_mse': 'mean',

                        # 'improve_percent1_mae': 'mean',
                        # 'improve_percent2_mae': 'mean',
                        # 'better_percent_mae': 'mean',
                        # 'draw_percent_mae': 'mean',
                        # 'improve_percent_in_better_mae': 'mean',
                        # 'improve_percent_in_worse_mae': 'mean',
                        # 'improve_percent_in_hard_mae': 'mean',
                        # 'improve_percent_in_medium_mae': 'mean',
                        # 'improve_percent_in_easy_mae': 'mean',
                        'improve_percent_mean_mae': 'mean',
                        'improve_percent_median_mae': 'mean',
                        'improve_percent_std_mae': 'mean',
                        'improve_percent_iqr_mae': 'mean',
                        'improve_percent_max_mae': 'mean',
                        'improve_percent_min_mae': 'mean',

                        'num_params': 'mean',
                        'Duration': 'mean',
                        'adapt_duration': 'mean',
                    }
                    selected_metrics = ['mse', 'mae']
                    perspectives = ['model_name', 'data_name', 'pred_len']
                    for perspective in perspectives:
                        summary_df = completed_exps.groupby(perspective).agg(agg_dict).reset_index()
                        summary_df_path = os.path.join(res_root_dir, f"_exp_summary_{perspective}.csv")
                        summary_df.to_csv(summary_df_path, index=False)
                        summary_df_short = summary_df.rename(columns=columns_short_dict)
                        _final_columns = [col for col in columns_short_dict.values() if col in summary_df_short.columns]
                        summary_df_short = summary_df_short[_final_columns]
                        logging.info(f"\n{summary_df_short.to_string()}")
                    # Aggregate the summary statistics
                    total_summary = completed_exps[list(agg_dict.keys())].agg(agg_dict)
                    total_summary_df = pd.DataFrame(total_summary).T  # Transpose to align the columns

                    # 保存结果到 CSV 文件
                    total_summary_df_path = os.path.join(res_root_dir, "_exp_summary_total.csv")
                    total_summary_df.to_csv(total_summary_df_path, index=False)
                    total_summary_df_short = total_summary_df.rename(columns=columns_short_dict)
                    _final_columns = [c for c in columns_short_dict.values() if c in total_summary_df_short.columns]
                    total_summary_df_short = total_summary_df_short[_final_columns]
                    logging.info(f"\n{total_summary_df_short.to_string()}")

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
    # 'Timer-LOTSA', 'Timer-UTSD'
    # 'Uni2ts-small', 'Uni2ts-base', 'Uni2ts-large'
    # 'Chronos-tiny'
    # 'Timer-LOTSA', 'Timer-UTSD', 'Uni2ts-base'
    # 'Timer-UTSD', 'Chronos-tiny', 'Uni2ts-base', 'Timer-LOTSA', 'Uni2ts-small', 'Uni2ts-large'
    # 'Uni2ts-base' seq长度192以上会内存不够 -> fix 96 好像就行了 -》nni意外
    # 'Unit2ts-large' 太消耗时间!!!-》batch解决了
    # 'Chronos-tiny' 还是慢 而且有各种问题..... # 可能是nan -》nan-inf解决了
    # Uni2ts-base
    # 'Timer-UTSD', 'MOIRAI-base', 'MOIRAI-small', 'MOIRAI-large', 'Timer-LOTSA', 'Chronos-tiny'
    model_names = ['Timer-UTSD', 'MOIRAI-base', 'Chronos-tiny', 'Timer-LOTSA', 'MOIRAI-small', 'MOIRAI-large'] \
        if not fast_mode else ['MOIRAI-small']  # FIXME:
    # 'ETTh1', 'Exchange', 'ETTh2', 'Electricity', 'ETTm1', 'Traffic', 'ETTm2', 'Weather'
    data_names = ['ETTh1', 'Exchange', 'ETTh2', 'Electricity', 'ETTm1', 'Traffic', 'ETTm2', 'Weather'] \
        if not fast_mode else ['ETTm1']  # FIXME:
    # pred_lens = [96, 192, 336, 720] if not fast_mode else [96]  # FIXME: 720长无意义能做好 96, 192
    pred_lens = [24, 48, 96, 192] if not fast_mode else [96]  # FIXME: Chronos和Decompose友好？！！

    # original_model_names = model_names
    # original_data_names = data_names
    # original_pred_lens = pred_lens

    # main()

    # 线性执行 multi 不同num_params 不同num_samples 不同seed 不同ablation
    # [25,37,50,75,100,175,250,375,500]
    # num_list = [25, 50, 100, 250, 500]
    # num_min, num_max, num_median = np.min(num_list), np.max(num_list), int(np.median(num_list))
    # FIXME: 补缺的
    # num_list = [37, 75, 175, 375]
    num_fix = 100

    ablations = [
        'none',  # 还是要ablation 中等实验P100S100不一定做过

        # 算子相关1 三大类
        # 'Context' # 类型1
        'Trimmer',
        'Sampler',  # ！
        'Aligner',  # ！
        # 'Range', # 类型2
        'Normalizer',  # ！
        'Warper',  # ！！！
        'Differentiator',
        # 'Anomaly' # 类型3
        'Inputer',  # ！
        'Denoiser',  # ！
        'Clipper',
        # 算子相关2
        'Pipeline',  # ！

        # 选择和排名有关 大框架有关
        'MultiMetric',
        'Stat',
        'Val',  # 'MultiPareto', 感觉跟Aug重了， 改了一点点
        'Aug',  # 可能需要细致的aug的消融
        'HPO',  # 不是主要贡献好像不重要。。。
    ]

    # seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #
    # _model_names = ['Timer-UTSD', 'MOIRAI-base', 'Chronos-tiny']  # LOTSA稳高，base居中
    # _data_names = ['ETTh1', 'ETTm1', 'Exchange', 'Electricity', 'Traffic', 'Weather']  # ETTh1难一点，ETTm2容易一点
    # _pred_lens = [96]

    # for seed in seed_list:
    #     for ablation in ablations:  # 消融实验
    #         main(_model_names, _data_names, _pred_lens,
    #              num_params=num_median, num_samples=num_median, ablation=ablation, seed=seed)

    # for seed in seed_list:
    #     # S从小到大: 中等规模的P
    #     for num_samples in num_list:
    #         if num_samples == num_fix:  # median已经做过了
    #             continue
    #         main(_model_names, _data_names, _pred_lens,
    #              num_params=num_fix, num_samples=num_samples, ablation='none', seed=seed)
    #
    # for seed in seed_list:
    #     # P从小到大: 中等规模的S
    #     for num_params in num_list:
    #         if num_params == num_fix:  # median已经做过了
    #             continue
    #         main(_model_names, _data_names, _pred_lens,
    #              num_params=num_params, num_samples=num_fix, ablation='none', seed=seed)

    # num_list = [1, 5, 10, 25, 37, 50, 75, 100, 175, 250, 375, 500]
    # for seed in seed_list:
    #     # P从小到大: 中等规模的S
    #     for num_params in num_list:
    #         if num_params == num_fix:  # median已经做过了
    #             continue
    #         main(_model_names, _data_names, _pred_lens,
    #              num_params=num_params, num_samples=num_fix, ablation='none', seed=seed)
    #
    # num_list = [1, 5, 10]
    # for seed in seed_list:
    #     # S从小到大: 中等规模的P
    #     for num_samples in num_list:
    #         if num_samples == num_fix:  # median已经做过了
    #             continue
    #         main(_model_names, _data_names, _pred_lens,
    #              num_params=num_fix, num_samples=num_samples, ablation='none', seed=seed)

    # for seed in seed_list:
    #     main(model_names, data_names, pred_lens,  # 全量实验
    #          num_params=500, num_samples=500, ablation='none', seed=seed)

    # MP 单进程实验 记录adptation时间：
    # _model_names = ['Timer-UTSD', 'MOIRAI-base', 'Chronos-tiny']
    # _data_names = ['ETTh1']
    # _pred_lens = [24, 48, 96, 192]
    # _seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # _max_processes = 8
    # _num_list = [500]
    # _fix_num = 100
    # P100S100
    # for seed in _seed_list:
    #     main(_model_names, _data_names, _pred_lens,
    #          num_params=_fix_num, num_samples=_fix_num, ablation='none', max_processes=_max_processes, seed=seed)
    # 固定P100，调整S
    # for seed in _seed_list:
    #     for num_samples in _num_list:
    #         if num_samples == _fix_num:
    #             continue
    #         main(_model_names, _data_names, _pred_lens,
    #              num_params=_fix_num, num_samples=num_samples, ablation='none', max_processes=_max_processes, seed=seed)
    # 固定S100，调整P
    # for seed in _seed_list:
    #     for num_params in _num_list:
    #         if num_params == _fix_num:
    #             continue
    #         main(_model_names, _data_names, _pred_lens,
    #              num_params=num_params, num_samples=_fix_num, ablation='none', max_processes=_max_processes, seed=seed)
    # num_samples_list = [500]
    # num_params_list = [500]
    # for seed in _seed_list:
    #     for num_samples in num_samples_list:
    #         for num_params in num_params_list:
    #             main(_model_names, _data_names, _pred_lens,
    #                  num_params=num_params, num_samples=num_samples, ablation='none', max_processes=_max_processes, seed=seed)
    
    parser = argparse.ArgumentParser(description='T3O')
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--model_names', type=str, default='Timer-UTSD', help="Model names (in string format or list of strings seperated by spaces)")
    parser.add_argument('--data_names', type=str, default='ETTh1', help='Data names (in string format or list of strings seperated by spaces)')
    parser.add_argument('--pred_lens', type=str, default='24 48 96 192', help='Prediction lengths (in string format or list of strings seperated by spaces)')
    parser.add_argument('--num_params', type=int, default=500, help='Number of parameters (in integer format)')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples (in integer format)')
    parser.add_argument('--ablation', type=str, default='none', help='Ablation mode (none, trim or clip)')
    parser.add_argument('--max_processes', type=int, default=1, help='Maximum number of processes to use')
    
    
    args = parser.parse_args()
    
    _model_names = args.model_names.split(' ')
    _data_names = args.data_names.split(' ')
    _pred_lens = [int(x) for x in args.pred_lens.split(' ')]
    # _num_params = [int(x) for x in args.num_params.split(' ')]
    # _num_samples = [int(x) for x in args.num_samples.split(' ')]
    # _ablation = args.ablation
    # _max_processes = args.max_processes
    main(
        _model_names,
        _data_names,
        _pred_lens,
        num_params=args.num_params,
        num_samples=args.num_samples,
        ablation=args.ablation,
        max_processes=args.max_processes,
        seed=args.seed
    )