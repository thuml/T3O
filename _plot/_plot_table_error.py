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

import os
from datetime import datetime

import pandas as pd

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

def generate_latex_table(df, file_name='latex_table3.tex'):
    with open(file_name, 'w') as f:
        # f.write('\\documentclass{article}\n')
        # f.write('\\usepackage{booktabs}\n')
        # f.write('\\usepackage{adjustbox}\n')
        # f.write('\\usepackage{multirow}\n')
        # f.write('\\usepackage{rotating}\n')
        # f.write('\\begin{document}\n')
        # f.write('\\begin{table*}[htbp]\n')
        # f.write('\\centering\n')
        # f.write('\\begin{adjustbox}{max width=\\textwidth}\n')
        # f.write('\\begin{tabular}{c|c|cc|cc|cc|cc|cc|cc}\n')
        # f.write('\\toprule\n')
        # f.write(
        #     '\\multicolumn{2}{c|}{\\textbf{Models}} & \\multicolumn{2}{c|}{\\textbf{Timer-UTSD}} & \\multicolumn{2}{c|}{\\textbf{Timer-LOTSA}} & \\multicolumn{2}{c|}{\\textbf{MOIRAI-small}} & \\multicolumn{2}{c|}{\\textbf{MOIRAI-base}} & \\multicolumn{2}{c|}{\\textbf{MOIRAI-large}} & \\multicolumn{2}{c}{\\textbf{CHRONOS-tiny}} \\\\\n')
        # f.write(
        #     '\\cmidrule(r){3-4} \\cmidrule(r){5-6} \\cmidrule(r){7-8} \\cmidrule(r){9-10} \\cmidrule(r){11-12} \\cmidrule(r){13-14}\n')
        # f.write(
        #     '\\multicolumn{2}{c|}{\\textbf{Metric Imp\\%}} & \\textbf{MSE\\%} & \\textbf{MAE\\%} & \\textbf{MSE\\%} & \\textbf{MAE\\%} & \\textbf{MSE\\%} & \\textbf{MAE\\%} & \\textbf{MSE\\%} & \\textbf{MAE\\%} & \\textbf{MSE\\%} & \\textbf{MAE\\%} & \\textbf{MSE\\%} & \\textbf{MAE\\%} \\\\\n')
        # f.write('\\midrule\n')

        datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Exchange', 'Traffic', 'Weather']
        for dataset in datasets:
            df_dataset = df[df['data_name'] == dataset]
            prediction_lengths = [24, 48, 96, 192]
            for i, pred_len in enumerate(prediction_lengths):
                df_pred_len = df_dataset[df_dataset['pred_len'] == pred_len]
                if i==0:
                    row = f"\\multirow{{4}}{{*}}{{{dataset}}} & {pred_len} "
                else:
                    row = f"& {pred_len} "
                for model in ['Timer-UTSD', 'Timer-LOTSA', 'MOIRAI-small', 'MOIRAI-base', 'MOIRAI-large',
                              'Chronos-tiny']:
                    # mse = df_pred_len[df_pred_len['model_name'] == model]['improve_percent_mean_mse'].values[0]
                    # mae = df_pred_len[df_pred_len['model_name'] == model]['improve_percent_mean_mae'].values[0]

                    mse = df_pred_len[df_pred_len['model_name'] == model]['our_mse'].values[0]
                    mae = df_pred_len[df_pred_len['model_name'] == model]['our_mae'].values[0]

                    mse_str = f"{mse:2.4f}"
                    mae_str = f"{mae:2.4f}"
                    # if mse > 0:
                    #     mse_str = f"\\textbf{{{mse_str}}}"
                    # if mae > 0:
                    #     mae_str = f"\\textbf{{{mae_str}}}"

                    row += f"& {mse_str} & {mae_str} "
                row += "\\\\\n"
                f.write(row)
            f.write('\\midrule\n') if dataset != datasets[-1] else None

        # f.write('\\bottomrule\n')
        # f.write('\\end{tabular}\n')
        # f.write('\\end{adjustbox}\n')
        # f.write(
        #     '\\caption{Multivariate results with different prediction lengths $O \\in \\{24, 48, 96, 192\\}$. We set the input length $I$ as 36 for ILI and 96 for the others. A lower MSE or MAE indicates a better prediction.}\n')
        # f.write('\\label{tab:multivariate_results}\n')
        # f.write('\\end{table*}\n')
        # f.write('\\end{document}\n')


def main():
    # 整合多次实验的结果数据，然后按照SEED聚合，最后封装成late格式
    # parent_dir_name = '/Users/cenzhiyao/Desktop/save/240612!!!-3'
    # children_dir_names = [
    #     "240707-001020-MF-P500S500-SD0",
    #     "240707-123137-MF-P500S500-SD1",
    #     "240707-181812-MF-P500S500-SD2",
    # ]

    parent_dir_name = '/Users/cenzhiyao/PycharmProjects/ts_adaptive_inference/new_moti'
    children_dir_names = get_dirs_from_range(
        parent_dir_name,
        '240715-003836-P500-S500-ABnone-SD0',
        '240717-010237-P500-S500-ABnone-SD8'
    )

    columns = [
        'data_name', 'model_name', 'pred_len',
        'improve_percent_mean_mse',
        'improve_percent_mean_mae',
        'org_mae', 'org_mse',
        'our_mae', 'our_mse',
    ]
    df_data = pd.DataFrame(columns=columns)

    for child_dir_name in children_dir_names:
        file_path = os.path.join(parent_dir_name, child_dir_name, '_experiment_status.csv')
        df = pd.read_csv(file_path)
        df_data = pd.concat([df_data, df], ignore_index=True)
    df_data = df_data[columns]
    df_data = df_data.groupby(['data_name', 'model_name', 'pred_len']).mean().reset_index()
    print(df_data.to_latex(index=False))
    generate_latex_table(df_data)


if __name__ == "__main__":
    main()
