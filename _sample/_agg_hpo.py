import itertools
import os


def main():
    res_root_dir = '~/PycharmProjects/ts_adaptive_inference/new_moti/240702-120526/'
    cur_tmp_dir = "./tmp"
    os.makedirs(cur_tmp_dir, exist_ok=True)
    model_names = ['Timer-LOTSA', 'Timer-UTSD', 'Uni2ts-base', 'Chronos-tiny', 'Uni2ts-small', 'Uni2ts-large']
    data_names = ['ETTh1', 'ETTh2', 'Exchange', 'Electricity', 'ETTm1', 'Weather', 'Traffic', 'ETTm2']
    pred_lens = [24, 48, 96, 192]

    # gpu_index_iter = itertools.cycle(gpu_indexes)
    for model_name, data_name, pred_len in itertools.product(model_names, data_names, pred_lens):
        res_dir = os.path.join(res_root_dir, f"{data_name}", f"{model_name}", f"pred_len-{pred_len}")
        hpo_path = os.path.join(res_dir, 'OT/train/_hpo_progress_plot.png')
        print(hpo_path)
        # 复制到cur_tmp_dir并改名为随机字符串
        os.system(f"cp {hpo_path} {cur_tmp_dir}")
        random_str = os.urandom(8).hex()
        new_hpo_path = os.path.join(cur_tmp_dir, f"{random_str}.png")
        os.system(f"mv {os.path.join(cur_tmp_dir, '_hpo_progress_plot.png')} {new_hpo_path}")
    # os.system(f"open {cur_tmp_dir}")
    return


if __name__ == '__main__':
    main()
