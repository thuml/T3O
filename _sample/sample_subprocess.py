import argparse
import subprocess
import concurrent.futures
import time
from datetime import datetime

import pandas as pd


def run_process(command):
    """启动一个进程并执行给定的命令"""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    return stdout, stderr


def main():
    parser = argparse.ArgumentParser(
        description='Concurrently run multiple python3 other processes with different arguments')
    parser.add_argument('--max_processes', type=int, default=2, help='Maximum number of concurrent processes')
    args = parser.parse_args()

    max_processes = args.max_processes

    # 要执行的命令和参数列表
    commands = [
        "python3 _other.py --arg1 value1 --arg2 value2",
        "python3 _other.py --arg1 value3 --arg2 value4",
        "python3 _other.py --arg1 value5 --arg2 value6",
        "python3 _other.py --arg1 value7 --arg2 value8",
        "python3 _other.py --arg1 value9 --arg2 value10",
        "python3 _other.py --arg1 value11 --arg2 value12",
        "python3 _other.py --arg1 value13 --arg2 value14",
        "python3 _other.py --arg1 value15 --arg2 value16",
        "python3 _other.py --arg1 value17 --arg2 value18",
        # 添加更多命令和参数...
    ]

    # 创建 DataFrame 用于记录实验状态和结果
    experiment_df = pd.DataFrame(columns=["Command", "Status", "Start Time", "End Time", "Duration", "Result"])
    # 先对 DataFrame 进行初始化
    for command in commands:
        experiment_df.loc[len(experiment_df)] = [command, "Pending", None, None, None, None]

    # 创建进程池执行器
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        # 提交每个进程的命令到进程池
        futures = [executor.submit(run_process, command) for command in commands]

        # 持续检查任务的状态和结果，直到所有任务完成
        while True:
            for future in futures:
                command_index = futures.index(future)
                command = commands[command_index]

                if future.running():
                    # 当对应任务的 Start Time 为空时，表示任务刚刚开始，填充实验开始时间和状态
                    if experiment_df.loc[experiment_df["Command"] == command, "Start Time"].values[0] is None:
                        # 记录实验开始时间和状态
                        start_time = datetime.now()
                        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
                        experiment_df.loc[experiment_df["Command"] == command, "Status"] = "Running"
                        experiment_df.loc[experiment_df["Command"] == command, "Start Time"] = start_time_str

                elif future.done():
                    # 当对应任务的 End Time 为空时，表示任务刚刚完成，填充实验结束时间、状态和结果
                    if experiment_df.loc[experiment_df["Command"] == command, "End Time"].values[0] is None:
                        start_time_str = experiment_df.loc[experiment_df["Command"] == command, "Start Time"].values[0]
                        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                        end_time = datetime.now()
                        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
                        duration = end_time - start_time
                        duration_str = str(duration)
                        stdout, stderr = future.result()
                        result = 0
                        # 记录实验结束时间、状态和结果
                        experiment_df.loc[experiment_df["Command"] == command, "Status"] = "Completed"
                        experiment_df.loc[experiment_df["Command"] == command, "End Time"] = end_time_str
                        experiment_df.loc[experiment_df["Command"] == command, "Duration"] = duration_str
                        experiment_df.loc[experiment_df["Command"] == command, "Result"] = result

            # 设置 pandas 显示选项，不限制行和列的展示数量
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)

            # 输出当前实验状态
            print("\nExperiment Status:")
            print(experiment_df)

            # 保存实验状态到 CSV 文件
            experiment_df.to_csv("experiment_status.csv", index=False)

            # 检查是否所有任务已完成
            if "Running" not in experiment_df["Status"].values:
                print("\nAll experiments completed.")
                break

            # 等待一段时间后再检查任务状态，避免过于频繁地检查
            time.sleep(5)


if __name__ == "__main__":
    main()
