import logging
import os
import sys
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import DataLoader

import matplotlib
import torch.cuda.amp as amp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from AnyTransform import ExpTimer
from AnyTransform.augmentor import Augmentor
from AnyTransform.dataset import get_dataset, CustomDataset
from transformers import AutoModelForCausalLM
from darts.models import ARIMA
from darts import TimeSeries

def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

class Arima:
    def __init__(self, model_name, ckpt_path, device='cpu'):
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.device = device
        self.patch_len = 96
        self.model = ARIMA()
    
    def forcast(self, data, pred_len):
        # import pdb;
        # pdb.set_trace()
        B,S,C = data.shape
        all_preds=[]
        for b in range(B):
            preds_for_b=[]
            for c in range(C):
                data_series = TimeSeries.from_values(data[b,:,c])
                self.model.fit(data_series)
                pred = self.model.predict(pred_len).values()
                pred = pred.reshape(1, pred_len, 1)
                preds_for_b.append(pred)
            preds_for_b = np.concatenate(preds_for_b, axis=2)
            all_preds.append(preds_for_b)
        final_preds = np.concatenate(all_preds, axis=0)
        final_preds = final_preds.reshape((B, pred_len, C))
        return final_preds


class TimerXL:
    def __init__(self, model_name, ckpt_path, device):
        self.model_name = model_name
        self.patch_len = 96
        self.device = self.choose_device(device)
        print(f'self.device: {self.device}')
        self.model = AutoModelForCausalLM.from_pretrained(
            #'/data/qiuyunzhong/Training-LTSM/checkpoints/models--thuml--timer-base/snapshots/35a991e1a21f8437c6d784465e87f24f5cc2b395',
            ckpt_path,
            trust_remote_code=True).to(self.device)
    
    def choose_device(self, device):
        if 'cpu' == device:
            return 'cpu'
        elif 'cuda' in device:
            idx = int(device.split(':')[-1])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            return 'cuda:0'
        else:
            raise ValueError(f'Unknown device: {device}')
    def forcast(self, data, pred_len):
        
        if len(data.shape) == 3:
            data = torch.tensor(data[:,:,-1]).squeeze().float().to(self.device)
            print('data.shape=', data.shape)
        pred = self.model.generate(data, max_new_tokens=pred_len)
        pred = pred.unsqueeze(2).detach().to('cpu').numpy()
        return pred


class Timer:
    def __init__(self, model_name, ckpt_path, device, args):
        # parser = argparse.ArgumentParser(description='TimesNet')
        #
        # # basic config
        # parser.add_argument('--task_name', type=str, default='large_finetune')
        # parser.add_argument('--seed', type=int, default=0)
        # parser.add_argument('--model', type=str, default='Timer')
        # parser.add_argument('--ckpt_path', type=str,
        #                     default='checkpoints/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt')
        # # model define
        # parser.add_argument('--patch_len', type=int, default=96)
        # parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
        # parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
        # parser.add_argument('--e_layers', type=int, default=8, help='num of encoder layers')
        # parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        # parser.add_argument('--factor', type=int, default=3, help='attn factor')
        # parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        # parser.add_argument('--activation', type=str, default='gelu', help='activation')
        # parser.add_argument('--output_attention', action='store_true')
        # # GPU
        # parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        # parser.add_argument('--gpu', type=int, default=0, help='gpu')
        # parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        # parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
        #
        # # args = parser.parse_args()
        #
        # args = argparse.Namespace(
        #     task_name='large_finetune',
        #     seed=0,
        #     model='Timer',
        #     ckpt_path=ckpt_path,
        #     patch_len=96,
        #     d_model=1024,
        #     n_heads=16,
        #     e_layers=8,
        #     d_ff=2048,
        #     factor=3,
        #     dropout=0.1,
        #     activation='gelu',
        #     output_attention=False,
        #     use_gpu=True,
        #     gpu=0,
        #     use_multi_gpu=False,
        #     devices='0,1,2,3'
        # )

        # _, args = parser.parse_known_args()  # 只解析默认的参数，而不是从命令行解析 ...

        # fix_seed = args.seed
        # random.seed(fix_seed)
        # torch.manual_seed(fix_seed)
        # np.random.seed(fix_seed)

        args.ckpt_path = ckpt_path
        assert 'cpu' == device or 'cuda' in device
        # args.use_gpu = True if 'cuda' in device else False
        # args.gpu = device.split(':')[-1] if 'cuda' in device else 0
        # args.gpu = '0'  # 在外设置了CUDA_VISIBLE_DEVICES -》好像不对，Timer还是需要设置cuda:x
        print(f'args.use_gpu={args.use_gpu}, args.gpu={args.gpu}')
        
        self.model_name = model_name
        self.args = args
        self.exp = ExpTimer(args)
        self.patch_len = self.args.patch_len  # 96

    def forcast(self, data, pred_len):
        # assert pred_len % self.patch_len == 0, f'pred_len={pred_len}'
        # step = pred_len // self.patch_len
        # # timer_scaler_flag=True之后，数据会被标准化！！！！
        # pred_total = self.exp.new_raw_inference(data, step, True)  # batch,time,feature
        # pred = pred_total[:, -pred_len:, :]
        # return pred
        _pred_total = self.exp.any_inference(data, pred_len)  # batch,time,feature
        pred = _pred_total[:, -pred_len:, :]
        return pred


import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import argparse
import random
import numpy as np


def get_model(model_name, device, args=None):
    if model_name == 'Timer-UTSD':
        model = Timer(model_name, '/data/qiuyunzhong/CKPT/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt', device, args)
    elif model_name == 'Timer-UTSD-PT24':
        model = Timer(model_name, '/data/qiuyunzhong/LTSM/checkpoints/25-03-01_17-41-52forecast_T3O_few_shot_Timer_T3O_ftS_sl672_ll576_pl24_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp/checkpoint.pth', device, args)
    elif model_name == 'Timer-UTSD-PT48':
        model = Timer(model_name, '/data/qiuyunzhong/LTSM/checkpoints/25-03-01_17-41-52forecast_T3O_few_shot_Timer_T3O_ftS_sl672_ll576_pl48_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp/checkpoint.pth', device, args)
    elif model_name == 'Timer-UTSD-PT96':
        model = Timer(model_name, '/data/qiuyunzhong/LTSM/checkpoints/25-03-01_17-41-51forecast_T3O_few_shot_Timer_T3O_ftS_sl672_ll576_pl96_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp/checkpoint.pth', device, args)
    elif model_name == 'Timer-UTSD-PT192':
        model = Timer(model_name, '/data/qiuyunzhong/LTSM/checkpoints/25-03-01_17-41-51forecast_T3O_few_shot_Timer_T3O_ftS_sl672_ll576_pl192_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp/checkpoint.pth', device, args)
    elif model_name == 'Timer-LOTSA':
        model = Timer(model_name, '/data/qiuyunzhong/CKPT/Large_timegpt_d1024_l8_p96_n64_new_full.ckpt', device, args)
    elif model_name == 'Timer-LOTSA-PT24':
        model = Timer(model_name, '/data/qiuyunzhong/LTSM/checkpoints/25-03-01_17-44-27forecast_T3O_few_shot_Timer_T3O_ftS_sl672_ll576_pl24_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp/checkpoint.pth', device, args)
    elif model_name == 'Timer-LOTSA-PT48':
        model = Timer(model_name, '/data/qiuyunzhong/LTSM/checkpoints/25-03-01_17-44-27forecast_T3O_few_shot_Timer_T3O_ftS_sl672_ll576_pl48_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp/checkpoint.pth', device, args)
    elif model_name == 'Timer-LOTSA-PT96':
        model = Timer(model_name, '/data/qiuyunzhong/LTSM/checkpoints/25-03-01_17-44-27forecast_T3O_few_shot_Timer_T3O_ftS_sl672_ll576_pl96_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp/checkpoint.pth', device, args)
    elif model_name == 'Timer-LOTSA-PT192':
        model = Timer(model_name, '/data/qiuyunzhong/LTSM/checkpoints/25-03-01_17-44-27forecast_T3O_few_shot_Timer_T3O_ftS_sl672_ll576_pl192_pl96_dm1024_nh8_el8_dl1_df2048_fc3_ebtimeF_dtTrue_Exp/checkpoint.pth', device, args)
    elif model_name == 'Timer1':
        model = Timer(model_name, '/data/qiuyunzhong/CKPT/Timer_forecast_1.0.ckpt', device, args)
    elif model_name == 'TimerXL':
        model = TimerXL(model_name, '/data/qiuyunzhong/Training-LTSM/checkpoints/models--thuml--timer-base/snapshots/35a991e1a21f8437c6d784465e87f24f5cc2b395', device)
    # elif model_name == '-PatchTST-UTSD':
    #     model = PatchTST('./-PatchTST/pretrain_checkpoints/UTSD-12G/ckpt_best.pth', device)
    elif model_name == 'MOIRAI-small':
        model = MOIRAI(model_name, './Uni2ts/ckpt/small', device)
    elif model_name == 'MOIRAI-base':
        model = MOIRAI(model_name, './Uni2ts/ckpt/base', device)
    elif model_name == 'MOIRAI-large':
        model = MOIRAI(model_name, './Uni2ts/ckpt/large', device)
    elif model_name == 'Chronos-tiny':
        model = Chronos(model_name, './Chronos/ckpt/tiny', device)
    elif model_name == 'Chronos-mini':
        model = Chronos(model_name, './Chronos/ckpt/mini', device)
    elif model_name == 'Chronos-small':
        model = Chronos(model_name, './Chronos/ckpt/small', device)
    elif model_name == 'Arima':
        model = Arima(model_name, None, 'cpu')
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model


class MOIRAI:  # 速度对batch敏感，几乎成倍增加时间
    def __init__(self, model_name, ckpt_path, device):
        # autocast:
        # cpu
        # none->float16->bfloat16
        # 0.17->0.14->0.143
        # gpu
        # none->float16->float32 (不支持bfloat)
        # 0.63->0.42->0.42->
        # -》float32
        self.model_name = model_name
        self.dtype = torch.float32  # 16节省内存 32最快？
        self.device = self.choose_device(device)
        # 10配128不错: org合理 our也有提升
        # for min: 20 128???
        # FIXME：sample=100没有提升了，10就可以
        self.num_samples = 20  # FIXME: 多次预测取median... 10效果不好试试30..
        # FIXME：！！！！针对small而言：64效果差0 128效果可10 8效果极好60 16效果极好50 （提升主要来源于波动减小稳定性高
        # 问题：patch=8 org效果太差离谱不能用，
        self.patch_size = 128  # FIXME: !!!!比auto的速度快很多！ # 96会有严重问题 必须是{8, 16, 32, 64, 128}中的一个 # 64效果不好试试128！！！！！！！！有效
        self.patch_len = self.patch_size  # FIXME:
        # 时间消耗：10：fast train: Timer-10s Uni2ts-large-100s Uni2ts-base-21s Uni2ts-large-7s
        self.module = MoiraiModule.from_pretrained(ckpt_path, local_files_only=True)
        # self.module = torch.jit.script(self.module)  # FIXME: 使用JIT加速  observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"] = None,
        logging.info(f'num_samples={self.num_samples}, patch_size={self.patch_size}')
        # small 100->0.3s, 30->0.18s, 10->0.15s, 5->0.13s, 1->0.14s

    def choose_device(self, device):
        if 'cpu' == device:
            return 'cpu'
        elif 'cuda' in device:
            idx = int(device.split(':')[-1])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            return 'cuda:0'
        else:
            raise ValueError(f'Unknown device: {device}')

    def forcast(self, data, pred_len):
        batch_size, seq_len, feature = data.shape
        # assert feature == 1, f'feature={feature}'

        # real_seq_l = len(data)
        real_seq_l = seq_len
        real_pred_l = pred_len

        # 重新拼接了同一个batch内的data！！！ 由此test_data的生成方式也会变！！！
        # _data = data.reshape(batch_size * real_seq_l)
        _data = data.reshape(batch_size * real_seq_l * feature)
        seq_with_zero_pred = np.concatenate([_data, np.zeros(real_pred_l)])
        date_range = pd.date_range(start='1900-01-01', periods=len(seq_with_zero_pred), freq='s')
        data_pd = pd.DataFrame(seq_with_zero_pred, index=date_range, columns=['target'])
        ds = PandasDataset(dict(data=data_pd))
        train, test_template = split(ds, offset=real_seq_l)
        test_data = test_template.generate_instances(
            prediction_length=real_pred_l,
            windows=batch_size,
            distance=real_seq_l,
        )

        with torch.no_grad(), amp.autocast(dtype=self.dtype):  # FIXME
            # with torch.no_grad():
            predictor = MoiraiForecast(
                module=self.module,
                prediction_length=real_pred_l,
                context_length=real_seq_l,
                patch_size=self.patch_size,  # FIXME: auto
                num_samples=self.num_samples,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            ).create_predictor(batch_size=batch_size, device=self.device)  # FIXME:batch_size=batch_size!!!
            forecasts = predictor.predict(test_data.input)
            forecast_list = list(forecasts)
        # assert len(forecast_list) == 1, f'len(forcast_list)={len(forecast_list)}'
        # forecast = forecast_list[0]
        # pred = forecast.quantile(0.5)  # median
        # assert len(pred) == real_pred_l, f'len(pred)={len(pred)}'
        # return pred
        assert len(forecast_list) == batch_size, f'len(forcast_list)={len(forecast_list)}'
        preds = np.array([forecast.quantile(0.5) for forecast in forecast_list])
        
        # ? Modification for covariate setting
        # preds = preds.reshape((batch_size, real_pred_l, 1))
        preds = preds.reshape((batch_size, real_pred_l, feature))
        return preds


from chronos import ChronosPipeline


class Chronos:  # pred较长时时间巨长...
    def __init__(self, model_name,ckpt_path, device):
        self.model_name = model_name
        self.device = self.choose_device(device)
        self.org_device = self.device
        self.ckpt_path = ckpt_path
        # gpu gpu autocast
        # none->float32->float16
        # 2.16->1.44->很慢->2.3
        # -> float32
        self.dtype = torch.float16  # 16节省内存 32最快？
        self.pipeline = ChronosPipeline.from_pretrained(
            self.ckpt_path,
            device_map=self.device,
            torch_dtype=self.dtype,
            # torch_dtype=torch.float64,  # 快
            # torch_dtype=torch.float32,  # 最快的！！！
            # torch_dtype=torch.bfloat16, # 不能用
            # torch_dtype=torch.float16,  # 更快
        )
        self.pipeline.model = self.pipeline.model.to(self.device)  # Ensure the model is on the correct device
        self.pipeline.model.eval()
        self.num_samples = 3  # FIXME: 多次预测取median... default=20 目测一个也能用 (多了CUDA内存爆炸
        # bfloat16,float16,float32,float64
        # 1->13s 7s 1.2s 1.7s
        # 3->19s 11s
        # 10->39s 23s

        # 相比Timer：17s->170s
        # 真相：因为Chronos内置Patch很短！！！！
        # 调整pred_len
        # 192->96->48->24->12->6
        # 2.3->1.2s->0.68->0.43->0.39->0.29
        self.patch_len = 512

    def reinit(self, device, dtype):
        self.device = self.choose_device(device)
        self.pipeline = None
        self.pipeline = ChronosPipeline.from_pretrained(
            self.ckpt_path,
            device_map=device,
            torch_dtype=dtype
        )
        self.pipeline.model.eval()

    def choose_device(self, device):
        if 'cpu' == device:
            return 'cpu'
        elif 'cuda' in device:
            idx = int(device.split(':')[-1])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            return 'cuda:0'
        else:
            raise ValueError(f'Unknown device: {device}')

    def forcast(self, data, pred_len):
        batch_size, seq_len, feature = data.shape
        assert feature == 1, f'feature={feature}'
        with torch.no_grad(), amp.autocast(dtype=self.dtype):  # FIXME:
            # with torch.no_grad():
            max_repeat = 5
            while max_repeat > 0:
                try:
                    if self.device != self.org_device:
                        logging.info(f'Chronos device changed, reinit...')
                        self.reinit(self.org_device, self.dtype)
                    # FIXME：既不能to dtype 也不能to device 都会报错
                    # data = torch.Tensor(data.reshape(batch_size, seq_len)).to(self.device)
                    # data = torch.Tensor(data.reshape(batch_size, seq_len)).to(self.dtype)
                    data = torch.Tensor(data.reshape(batch_size, seq_len))
                    forecast = self.pipeline.predict(
                        context=data,
                        prediction_length=pred_len,
                        num_samples=self.num_samples,
                        limit_prediction_length=False,
                    )
                    break
                except Exception as e:
                    logging.error(e)
                    logging.info(f'Chronos predict failed, max_repeat={max_repeat}, reinit...')
                    time.sleep(3)
                    # device = 'cuda:0' if max_repeat != 1 else 'cpu'
                    # dtype = random.choice([torch.float16, torch.float32, torch.float64])
                    device, dtype = self.device, self.dtype
                    logging.info(f'device={device}, dtype={dtype}')
                    try:
                        self.reinit(device, dtype)  # 也会失败
                    except Exception as e:
                        logging.error(e)
                        logging.info(f'Chronos reinit failed, max_repeat={max_repeat}, reinit...')
                    max_repeat -= 1
                    if max_repeat == 0:
                        raise ValueError(f'Chronos predict failed, with error: {e}')
            assert forecast.shape == (batch_size, self.num_samples, pred_len), f'forecast.shape={forecast.shape}'
            preds = np.median(forecast.numpy(), axis=1).reshape((batch_size, pred_len, 1))
            return preds


def time_start():
    return time.time()


def log_time_delta(t, event_name):
    d = time.time() - t
    print(f"{event_name} time: {d}")


# Example usage
if __name__ == "__main__":
    seq_len = 96 * 4
    pred_len = 192
    # dataset = get_dataset('ETTh1')
    # dataset = get_dataset('ETTm1')
    # dataset = get_dataset('Exchange')
    dataset = get_dataset('Weather')
    # seq = dataset.np_data_dict['OT'][dataset.train_start:dataset.train_end][:seq_len]
    # truth_total = dataset.np_data_dict['OT'][dataset.train_start:dataset.train_end][:seq_len + pred_len]

    mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
        'train', 'OT', seq_len, Augmentor('none', 'fix'), 10, 10
    custom_dataset = CustomDataset(dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
    idx, history, label = next(iter(dataloader))  # batch, time, feature
    # history, label = history.numpy(), label.numpy()
    # 从4维
    history = history.reshape(batch_size, seq_len, 1)
    label = label.reshape(batch_size, pred_len, 1)

    # 对每个 batch 分别计算均值并进行缩放
    history_transformed = np.zeros_like(history)
    label_transformed = np.zeros_like(label)
    scalers = []
    for i in range(batch_size):
        scaler = StandardScaler()
        history_batch = history[i].numpy().reshape(-1, 1)
        label_batch = label[i].numpy().reshape(-1, 1)
        # 对每个 batch 的数据进行缩放
        history_transformed[i] = scaler.fit_transform(history_batch).reshape(seq_len, 1)
        label_transformed[i] = scaler.transform(label_batch).reshape(pred_len, 1)
        scalers.append(scaler)  # 保存每个 batch 的 scaler 以备后用
    # 将数据转换回原始类型
    history = np.array(history_transformed)
    label = np.array(label_transformed)

    seqs = history.copy()
    print("seqs.shape", seqs.shape)

    device = "cpu" if sys.platform == 'darwin' else 'cuda:0'

    # model = get_model('Timer-LOTSA', device)
    # model = get_model('Chronos-tiny', device)
    model = get_model('Timer-UTSD', device)
    # model = get_model('Uni2ts-small', device)
    # model = get_model('MOIRAI-small', device)
    t = time_start()
    preds = model.forcast(seqs, pred_len)
    log_time_delta(t, 'Preprocess')
    print("preds.shape", preds.shape)

    # 画个图吧
    # 把batch内的数据画在不同的子图上
    plt.figure(figsize=(12, 6))
    for i in range(batch_size):
        plt.subplot(batch_size, 1, i + 1)
        plt.plot(np.arange(seq_len + pred_len), np.concatenate([seqs[i], preds[i]]), label='pred')
        plt.plot(np.arange(seq_len + pred_len), np.concatenate([history[i], label[i]]), label='truth')
        plt.legend()
        from Timer.utils.metrics import metric

        mae, mse, rmse, mape, mspe = metric(preds[i], label[i])
        plt.title(f"mae={mae:.4f}, mse={mse:.4f}, rmse={rmse:.4f}, mape={mape:.4f}, mspe={mspe:.4f}")
    plt.show()
