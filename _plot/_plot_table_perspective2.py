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

def generate_latex_table(df, file_name='latex_table_perspective.tex'):
    with open(file_name, 'w') as f:
        # f.write('\\begin{table}[htbp]\n')
        # f.write('    \\centering\n')
        # f.write('    \\begin{adjustbox}{max width=\\textwidth}\n')
        # f.write('    \\begin{tabular}{c|c|cc|cc}\n')
        # f.write('        \\toprule\n')
        # f.write(
        #     '        \\multicolumn{2}{c|}{\\textbf{Statistics}} & \\multicolumn{2}{c|}{\\textbf{Mean}} & \\multicolumn{2}{c}{\\textbf{Median}} \\\\\n')
        # f.write('        \\cmidrule(r){3-4} \\cmidrule(r){5-6}\n')
        # f.write(
        #     '        \\multicolumn{2}{c|}{\\textbf{Metric Imp\\%}} & \\textbf{Mse\\%} & \\textbf{Mae\\%} & \\textbf{Mse\\%} & \\textbf{Mae\\%} \\\\\n')
        # f.write('        \\midrule\n')

        # models = df['model_name'].unique()
        models = ['Timer-UTSD', 'Timer-LOTSA', 'MOIRAI-small', 'MOIRAI-base', 'MOIRAI-large', 'Chronos-tiny']
        for i, item in enumerate(models):
            # item = 'Moirai' if item == 'MOIRAI' else item
            filtered_data = df[df['model_name'] == item]

            org_mse_mean = filtered_data['org_mean_mse'].mean()



            mse_mean = filtered_data['improve_percent_mean_mse'].mean()
            mae_mean = filtered_data['improve_percent_mean_mae'].mean()
            mse_median = filtered_data['improve_percent_median_mse'].median()
            mae_median = filtered_data['improve_percent_median_mae'].median()
            mse_std = filtered_data['improve_percent_std_mse'].mean()
            mae_std = filtered_data['improve_percent_std_mae'].mean()
            mse_iqr = filtered_data['improve_percent_iqr_mse'].mean()
            mae_iqr = filtered_data['improve_percent_iqr_mae'].mean()
            item = item.replace('MOIRAI', 'Moirai')
            if i == 0:
                f.write(
                    f'        \\multirow{{{len(models)}}}{{*}}{{\\textbf{{Model}}}} & {item} & {mse_mean:.2f}\\% & {mae_mean:.2f}\\% & {mse_median:.2f}\\% & {mae_median:.2f}\\% & {mse_std:.2f}\\% & {mae_std:.2f}\\% & {mse_iqr:.2f}\\% & {mae_iqr:.2f}\\% \\\\\n')
            else:
                # 添加%符号在latex中
                f.write(
                    f'         & {item} & {mse_mean:.2f}\\% & {mae_mean:.2f}\\% & {mse_median:.2f}\\% & {mae_median:.2f}\\% & {mse_std:.2f}\\% & {mae_std:.2f}\\% & {mse_iqr:.2f}\\% & {mae_iqr:.2f}\\% \\\\\n')
        f.write('        \\midrule\n')

        data_names = df['data_name'].unique()
        for i, item in enumerate(data_names):
            filtered_data = df[df['data_name'] == item]
            mse_mean = filtered_data['improve_percent_mean_mse'].mean()
            mae_mean = filtered_data['improve_percent_mean_mae'].mean()
            mse_median = filtered_data['improve_percent_median_mse'].median()
            mae_median = filtered_data['improve_percent_median_mae'].median()
            mse_std = filtered_data['improve_percent_std_mse'].mean()
            mae_std = filtered_data['improve_percent_std_mae'].mean()
            mse_iqr = filtered_data['improve_percent_iqr_mse'].mean()
            mae_iqr = filtered_data['improve_percent_iqr_mae'].mean()

            if i == 0:
                f.write(
                    f'        \\multirow{{{len(data_names)}}}{{*}}{{\\textbf{{Data}}}} & {item} & {mse_mean:.2f}\\% & {mae_mean:.2f}\\% & {mse_median:.2f}\\% & {mae_median:.2f}\\% & {mse_std:.2f}\\% & {mae_std:.2f}\\% & {mse_iqr:.2f}\\% & {mae_iqr:.2f}\\% \\\\\n')
            else:
                f.write(
                    f'         & {item} & {mse_mean:.2f}\\% & {mae_mean:.2f}\\% & {mse_median:.2f}\\% & {mae_median:.2f}\\% & {mse_std:.2f}\\% & {mae_std:.2f}\\% & {mse_iqr:.2f}\\% & {mae_iqr:.2f}\\% \\\\\n')
        f.write('        \\midrule\n')

        tasks = df['pred_len'].unique()
        for i, item in enumerate(tasks):
            filtered_data = df[df['pred_len'] == item]
            mse_mean = filtered_data['improve_percent_mean_mse'].mean()
            mae_mean = filtered_data['improve_percent_mean_mae'].mean()
            mse_median = filtered_data['improve_percent_median_mse'].median()
            mae_median = filtered_data['improve_percent_median_mae'].median()
            mse_std = filtered_data['improve_percent_std_mse'].mean()
            mae_std = filtered_data['improve_percent_std_mae'].mean()
            mse_iqr = filtered_data['improve_percent_iqr_mse'].mean()
            mae_iqr = filtered_data['improve_percent_iqr_mae'].mean()

            if i == 0:
                f.write(
                    f'        \\multirow{{{len(tasks)}}}{{*}}{{\\textbf{{Task}}}} & {item} & {mse_mean:.2f}\\% & {mae_mean:.2f}\\% & {mse_median:.2f}\\% & {mae_median:.2f}\\% & {mse_std:.2f}\\% & {mae_std:.2f}\\% & {mse_iqr:.2f}\\% & {mae_iqr:.2f}\\% \\\\\n')
            else:
                f.write(
                    f'         & {item} & {mse_mean:.2f}\\% & {mae_mean:.2f}\\% & {mse_median:.2f}\\% & {mae_median:.2f}\\% & {mse_std:.2f}\\% & {mae_std:.2f}\\% & {mse_iqr:.2f}\\% & {mae_iqr:.2f}\\% \\\\\n')

        #          \midrule
        #          \multicolumn{2}{c|}{\textbf{Average}} & 10.99 & 10.23 & 8.35 & 9.44 \\

        # f.write('        \\midrule\n')
        # mse_mean = df['improve_percent_mean_mse'].mean()
        # mae_mean = df['improve_percent_mean_mae'].mean()
        # mse_median = df['improve_percent_median_mse'].median()
        # mae_median = df['improve_percent_median_mae'].median()
        # mse_std = df['improve_percent_std_mse'].mean()
        # mae_std = df['improve_percent_std_mae'].mean()
        # mse_iqr = df['improve_percent_iqr_mse'].mean()
        # mae_iqr = df['improve_percent_iqr_mae'].mean()
        # f.write(f'        \\multicolumn{{2}}{{c|}}{{\\textbf{{Average}}}} & \\textbf{{{mse_mean:.2f}}}\\% & \\textbf{{{mae_mean:.2f}}}\\% & \\textbf{{{mse_median:.2f}}}\\% & \\textbf{{{mae_median:.2f}}}\\% & \\textbf{{{mse_std:.2f}}}\\% & \\textbf{{{mae_std:.2f}}}\\% & \\textbf{{{mse_iqr:.2f}}}\\% & \\textbf{{{mae_iqr:.2f}}}\\% \\\\\n')
        #


        # f.write('        \\bottomrule\n')
        # f.write('    \\end{tabular}\n')
        # f.write('    \\end{adjustbox}\n')
        # f.write('    \\caption{Statistic Metrics of Improvement Percentage for Different Models, Data, and Tasks.}\n')
        # f.write('    \\label{tab:stat_metrics}\n')
        # f.write('\\end{table}\n')


def main():
    # Load the data
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
        'improve_percent_median_mse',
        'improve_percent_median_mae',
        'improve_percent_std_mse',
        'improve_percent_std_mae',
        'improve_percent_iqr_mse',
        'improve_percent_iqr_mae',
    ]
    df_data = pd.DataFrame(columns=columns)

    for child_dir_name in children_dir_names:
        file_path = os.path.join(parent_dir_name, child_dir_name, '_experiment_status.csv')
        df = pd.read_csv(file_path)
        df_data = pd.concat([df_data, df], ignore_index=True)
    df_data = df_data[columns]
    df_data = df_data.groupby(['data_name', 'model_name', 'pred_len']).mean().reset_index()

    print(df_data)

    # Generate LaTeX table
    generate_latex_table(df_data)


if __name__ == "__main__":
    main()
