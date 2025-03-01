import time
from typing import Tuple, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import ndarray

from ..data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from ..utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from ..utils.tools import EarlyStopping, visual, LargeScheduler


# mpl.use('TkAgg')
warnings.filterwarnings('ignore')


class Exp_Large_Few_Shot_Roll_Demo(Exp_Basic):

    def __init__(self, args):
        super(Exp_Large_Few_Shot_Roll_Demo, self).__init__(args)
    
    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args)
            model = DDP(model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        if self.args.use_weight_decay:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def vali(self, process, vali_data, vali_loader, criterion, epoch=0, flag='vali'):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                batch_x = process.preprocess(batch_x, target_column='OT', pred_len=self.args.pred_len, patch_len=self.args.patch_len)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                if self.args.output_attention:
                    # output used to calculate loss misaligned patch_len compared to input
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    # only use the forecast window to calculate loss
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = process.post_process(outputs, pred_len=self.args.pred_len)
                if self.args.use_ims:
                    pred = outputs[:, -self.args.seq_len:, :]
                    true = batch_y
                    if flag == 'vali':
                        loss = criterion(pred, true)
                    elif flag == 'test':  # in this case, only pred_len is used to calculate loss
                        pred = pred[:, -self.args.pred_len:, :]
                        true = true[:, -self.args.pred_len:, :]
                        loss = criterion(pred, true)
                else:
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])
                
                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                torch.cuda.empty_cache()
        
        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss
    
    def finetune(self, setting, process):
        finetune_data, finetune_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        test_data, test_loader = data_provider(self.args, flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)
        
        time_now = time.time()
        
        train_steps = len(finetune_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        scheduler = LargeScheduler(self.args, model_optim)
        
        for epoch in range(self.args.finetune_epochs):
            iter_count = 0
            
            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")
            
            self.model.train()
            epoch_time = time.time()
            
            print("Step number per epoch: ", len(finetune_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(finetune_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x = process.preprocess(batch_x, target_column='OT', pred_len=self.args.pred_len, patch_len=self.args.patch_len)
                # batch_x = torch.from_numpy(batch_x).float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = process.post_process(outputs, pred_len=self.args.pred_len)
                # outputs = torch.from_numpy(outputs).float().to(self.device)
                
                if self.args.use_ims:
                    # output used to calculate loss misaligned patch_len compared to input
                    loss = criterion(outputs[:, -self.args.seq_len:, :], batch_y)
                else:
                    # only use the forecast window to calculate loss
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])
                
                loss_val += loss
                count += 1
                
                if i % 50 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
                torch.cuda.empty_cache()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()
            
            vali_loss = self.vali(process, vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(process, test_data, test_loader, criterion, flag='test')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.schedule_epoch(epoch)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    
    def test(self, setting, process, test=0):
        if not self.args.is_training and self.args.finetuned_ckpt_path:
            print('loading model: ', self.args.finetuned_ckpt_path)
            if self.args.finetuned_ckpt_path.endswith('.pth'):
                sd = torch.load(self.args.finetuned_ckpt_path, map_location="cpu")
                self.model.load_state_dict(sd, strict=True)
            
            elif self.args.finetuned_ckpt_path.endswith('.ckpt'):
                if self.args.use_multi_gpu:
                    sd = torch.load(self.args.finetuned_ckpt_path, map_location="cpu")["state_dict"]
                    sd = {'module.' + k: v for k, v in sd.items()}
                    self.model.load_state_dict(sd, strict=True)
                else:
                    sd = torch.load(self.args.finetuned_ckpt_path, map_location="cpu")["state_dict"]
                    self.model.load_state_dict(sd, strict=True)
            else:
                raise NotImplementedError
        
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        target_root_path = self.args.root_path
        target_data_path = self.args.data_path
        target_data = self.args.data
        
        print("=====================Testing: {}=====================".format(target_root_path + target_data_path))
        print("=====================Demo: {}=====================".format(target_root_path + target_data_path))
        test_data, test_loader = data_provider(self.args, flag='test')
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/' + target_data_path + '/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        # device = torch.device('cuda:{}'.format(self.args.local_rank)) if self.args.use_gpu else torch.device('cpu')
        # mae_val = torch.tensor(0., device=device)
        # mse_val = torch.tensor(0., device=device)
        # count = torch.tensor(1e-5, device=device)
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x = process.preprocess(batch_x, target_column='OT', pred_len=self.args.pred_len, patch_len=self.args.patch_len)
                
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                inference_steps = self.args.output_len // self.args.pred_len
                dis = self.args.output_len - inference_steps * self.args.pred_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                # encoder - decoder
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j - 1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)
                    
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = process.post_process(outputs, pred_len=self.args.pred_len)
                    outputs = outputs[:, -self.args.pred_len:, -1:]
                    f_dim = -1 if self.args.features == 'MS' else 0
                    pred_y.append(outputs[:, -self.args.pred_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-dis, :]
                
                # batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = pred_y[:, -self.args.pred_len:, :].detach().cpu()
                batch_y = batch_y.detach().cpu()
                
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.post_process(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.post_process(batch_y.squeeze(0)).reshape(shape)
                
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                
                pred = outputs
                true = batch_y
                
                preds.append(pred)
                trues.append(true)
                
                # if i % 10 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)
                #
                #     dir_path = folder_path + f'{self.args.output_len}/'
                #     if not os.path.exists(dir_path):
                #         os.makedirs(dir_path)
                #     # np.save(os.path.join(dir_path, f'gt_{i}.npy'), np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0))
                #     # np.save(os.path.join(dir_path, f'pd_{i}.npy'), np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0))
                #     np.savez(os.path.join(dir_path, f'res_{i}.npz'), groundtruth=gt, predict=pd)
                #     print(os.path.join(dir_path, f'res_{i}.npz'), "saved")
                
                # if self.args.use_multi_gpu:
                #     visual(gt, pd, os.path.join(dir_path, f'{i}_{self.args.local_rank}.pdf'))
                # else:
                #     visual(gt, pd, os.path.join(dir_path, f'{i}_.pdf'))
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        
        # # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        return

    def inference(self, data: ndarray, pred_len: int = 96) -> tuple[str, Any]:
        # 以该函数被调用的时间为准，创建一个文件夹
        folder_path = './inference_results/' + datetime.now().strftime("%Y%m%d_%H%M%S%f") + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        # 存储一个data的副本
        original_data = data.copy()  # [seq_len, 1]

        patch_len = self.args.patch_len
        # data的形状为[L, 1]，将L补齐到patch_len的整数倍
        seq_len = data.shape[0]
        pad_len = (patch_len - seq_len % patch_len) % patch_len
        if seq_len % patch_len != 0:
            data = np.concatenate((np.zeros((pad_len, 1)), data), axis=0)

        # 在data前面挤压一维，变成[1, L, 1]
        data = data[np.newaxis, :, :]  # [1, seq_len + pad_len, 1]
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        inference_steps = pred_len // patch_len
        dis = inference_steps * patch_len - pred_len
        if dis != 0:
            inference_steps += 1
            dis = dis + patch_len

        # encoder - decoder
        self.model.eval()
        with torch.no_grad():
            for j in range(inference_steps):
                outputs = self.model(data, None, None, None)
                data = torch.cat([data, outputs[:, -patch_len:, :]], dim=1)

        if dis != 0:
            data = data[:, :-dis, :]

        data = data.detach().cpu().numpy()
        data = data.squeeze(0)  # [seq_len + pad_len + pred_len, 1]

        pred_data = data[pad_len:, 0]

        # 保存原始数据和预测数据
        np.save(os.path.join(folder_path, 'original_data.npy'), original_data)
        np.save(os.path.join(folder_path, 'pred_data.npy'), pred_data)

        plt.figure(figsize=((seq_len + pred_len) // patch_len * 5, 5))

        # 绘制图像
        plt.plot(np.arange(seq_len + pred_len), pred_data, label='Prediction', c='dodgerblue', linewidth=2)
        plt.plot(np.arange(seq_len), original_data, label='GroundTruth', c='tomato', linewidth=2)

        # 添加图例
        plt.legend()

        # 保存图片
        plt.savefig(os.path.join(folder_path, 'inference_result.pdf'), bbox_inches='tight')

        return folder_path, pred_data

    def new_raw_inference(self, data, inference_step, timer_scaler_flag) -> np.ndarray:
        # assert data_copy.shape[0] == 1 and data_copy.shape[2] == 1, \
        #     f'data_copy shape {data_copy.shape} is not [1,seq_len,1]'
        # assert data_copy.shape[1] % self.args.patch_len == 0, \
        #     f'seq_len {data_copy.shape[1]} is not a multiple of patch_len {self.args.patch_len}'
        # assert inference_step > 0, f'inference_step {inference_step} should be greater than 0'

        # data_copy: batch,time,feature
        batch_size, seq_len, feature = data.shape
        assert feature == 1, f'feature {feature} should be 1'
        assert seq_len % self.args.patch_len == 0, \
            f'seq_len {seq_len} is not a multiple of patch_len {self.args.patch_len}'
        assert inference_step > 0, f'inference_step {inference_step} should be greater than 0'

        # FIXME: 16节省内存 32最快 # 32-> (1000, 1056, 1) 会崩 (800, 672, 1) 也会？？？
        dtype = torch.float32  # 16无法在cpu上运行

        seq = torch.tensor(data, dtype=dtype).to(self.device)
        patch_len = self.args.patch_len

        # 注意！PatchTST本身不是auto_regressive：输入336和输出长度96都是固定的！

        self.model.eval()
        # gpu autocast
        # none->float16->float32
        # 32->36->53
        with torch.no_grad() and torch.cuda.amp.autocast(dtype=dtype):
            # with torch.no_grad():
            for j in range(inference_step):
                # FIXME: timer_scaler_flag 决定了是否使用内置的scaler！！！
                if timer_scaler_flag:
                    outputs = self.model(seq, None, None, None)
                else:
                    outputs = self.model.raw_forcast(seq)
                seq = torch.cat([seq, outputs[:, -patch_len:, :]], dim=1)
        pred_total = seq.detach().cpu().numpy()
        assert pred_total.shape[0] == batch_size, f'batch size {pred_total.shape[0]} should be {batch_size}'
        assert pred_total.shape[2] == 1, f'feature {pred_total.shape[2]} should be 1'
        return pred_total

    def any_inference(self, data: ndarray, pred_len: int = 96) -> np.ndarray:
        # 有norm和zero-pad的inference（但不写本地
        # 以该函数被调用的时间为准，创建一个文件夹

        batch_size, seq_len, feature = data.shape
        assert feature == 1, f'feature {feature} should be 1'
        # assert seq_len % self.args.patch_len == 0, \
        #     f'seq_len {seq_len} is not a multiple of patch_len {self.args.patch_len}'

        # FIXME: 16节省内存 32最快 # 32-> (1000, 1056, 1) 会崩 (800, 672, 1) 也会？？？
        dtype = torch.float32  # 16无法在cpu上运行

        patch_len = self.args.patch_len
        pad_len = (patch_len - seq_len % patch_len) % patch_len
        if seq_len % patch_len != 0:
            data = np.concatenate((np.zeros((batch_size, pad_len, 1)), data), axis=1)
        data = torch.tensor(data, dtype=dtype).to(self.device)
        inference_step = pred_len // patch_len
        dis = inference_step * patch_len - pred_len
        if dis != 0:
            inference_step += 1
            dis = dis + patch_len

        seq = torch.tensor(data, dtype=dtype).to(self.device)
        assert seq.shape[1] % patch_len == 0, f'seq_len {seq.shape[1]} is not a multiple of patch_len {patch_len}'

        self.model.eval()
        # gpu autocast
        # none->float16->float32
        # 32->36->53
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            # with torch.no_grad():
            for j in range(inference_step):
                outputs = self.model(seq, None, None, None)  # 使用了内置的scaler！！！
                seq = torch.cat([seq, outputs[:, -patch_len:, :]], dim=1)
        if dis != 0:
            seq = seq[:, :-dis, :]  # 去掉pred多余的部分
        _pred_total = seq.detach().cpu().numpy()
        pred_total = _pred_total[:, pad_len:, :]
        assert pred_total.shape[1] == seq_len + pred_len, \
            f'pred_len {pred_total.shape[1]} should be seq_len {seq_len} + pred_len {pred_len}'
        assert pred_total.shape[0] == batch_size, f'batch size {pred_total.shape[0]} should be {batch_size}'
        assert pred_total.shape[2] == 1, f'feature {pred_total.shape[2]} should be 1'
        return pred_total
