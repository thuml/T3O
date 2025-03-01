import argparse
import os
from datetime import datetime
import torch

from exp.exp_large_few_shot_roll_demo import Exp_Large_Few_Shot_Roll_Demo
import random
import numpy as np
import torch.distributed as dist

from utils.tools import HiddenPrints
if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./ckpt/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)# Timer1 param
    

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='None',
                        help='task name, options:[partial_train, zero_shot, decompose]')

    parser.add_argument('--partial_part', type=int, default=0, help='partial_train')
    parser.add_argument('--random_train', type=bool, default=False, help='random_train')
    parser.add_argument('--channel_independent', type=bool, default=False, help='channel_independent')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/ETT-small/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--target_data', type=str, default='custom', help='target dataset type')
    parser.add_argument('--target_root_path_list', nargs="+", help="the root path of the data file")
    parser.add_argument('--target_data_path_list', nargs="+", help="the data file")
    parser.add_argument('--target_data_list', nargs="+", help="the stride of the data file")

    parser.add_argument('--exchange_attention', type=bool, default=False, help='use gpu')

    parser.add_argument('--root_path_list', nargs="+", help="the root path of the data file")
    parser.add_argument('--data_path_list', nargs="+", help="the data file")
    parser.add_argument('--stride_list', nargs="+", help="the stride of the data file")
    parser.add_argument('--stride', type=int, default=1, help='stride')

    parser.add_argument('--decompose_order', type=str, required=False, default='last', help='decompose order [last first]')
    parser.add_argument('--decompose_strategy', type=str, required=False, default='one', help='decompose strategy [one, two, cat]')

    parser.add_argument('--loss_type', type=str, required=False, default='mse', help='decompose order [mse sdb]')

    parser.add_argument('--long_embed', type=bool, default=False, help='long_embed')
    parser.add_argument('--decompose_type', type=str, required=False, default='default', help='decompose order [default stl]')
    parser.add_argument('--ckpt_path', type=str, default='', help='ckpt file')
    parser.add_argument('--ckpt_output_path', type=str, default='./tmp_ckpt/large_debug_default.pth', help='ckpt file')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--finetune_rate', type=float, default=0.1, help='finetune ratio')
    # htrm
    parser.add_argument('--patch_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--subset_rand_ratio', type=float, default=1, help='mask ratio')
    parser.add_argument('--subset_rand_rand_ratio', type=float, default=1, help='mask ratio')


    parser.add_argument('--data_type', type=str, default='custom', help='data_type')
    # subset_ratio
    parser.add_argument('--subset_ratio', type=float, default=1, help='subset_ratio')
    # split_ratio
    parser.add_argument('--split', type=float, default=0.9, help='split_ratio')
    parser.add_argument('--decay_fac', type=float, default=0.75)

    #cos_warm_up_steps cos_max_decay_steps cos_max cos_min
    parser.add_argument('--cos_warm_up_steps', type=int, default=100)
    parser.add_argument('--cos_max_decay_steps', type=int, default=60000)
    parser.add_argument('--cos_max_decay_epoch', type=int, default=10)

    parser.add_argument('--cos_max', type=float, default=1e-4)
    parser.add_argument('--cos_min', type=float, default=2e-6)

    # freeze
    parser.add_argument('--freeze', type=int, default=0, help='use_post_data')
    # ['norm', 'value_embedding', 'head'] ['norm'] ['head']
    parser.add_argument('--unfrozen_list', nargs="+", default=['norm', 'value_embedding', 'proj.'], help="the data file")

    # weight decay
    parser.add_argument('--use_weight_decay', type=int, default=0, help='use_post_data')
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # few shot
    parser.add_argument('--use_post_data', type=int, default=0, help='use_post_data')
    parser.add_argument('--roll', action='store_true', help='roll', default=False)
    parser.add_argument('--output_len', type=int, default=96, help='output len')
    # train_test
    parser.add_argument('--train_test', type=int, default=1, help='train_test')

    parser.add_argument('--is_finetuning', type=int, default=1, help='status')
    parser.add_argument('--finetuned_ckpt_path', type=str, default='', help='ckpt file')

    # train_offset
    parser.add_argument('--train_offset', type=int, default=0, help='train_offset')

    # alumina
    # parser.add_argument('--trainset_csv_path', type=str, required=True, help='')
    # parser.add_argument('--validset_csv_path', type=str, required=True, help='')
    # parser.add_argument('--testset_csv_path', type=str, required=True, help='')
    parser.add_argument('--datetime_col', type=str, help='')
    parser.add_argument('--feature_cols', type=str, help='')
    parser.add_argument('--target_cols', type=str, help='')
    parser.add_argument('--timestamp_feature', type=str, help='')

    # Hosted Demo
    parser.add_argument('--show_demo', action='store_true', help='show demo', default=False)


    args = parser.parse_args()
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))  # number of nodes
        rank = int(os.environ.get("RANK", "0"))  # node id
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()  # gpus per node
        args.local_rank = local_rank
        print('ip: {}, port: {}, hosts: {}, rank: {}, local_rank: {}, gpus: {}'.format(ip, port, hosts, rank, local_rank, gpus))
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
        print('init_process_group finished')
        torch.cuda.set_device(local_rank)

    if args.task_name == 'large_finetune':
        if args.roll:
            if args.show_demo:
                Exp = Exp_Large_Few_Shot_Roll_Demo
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


    with HiddenPrints(int(os.environ.get("LOCAL_RANK", "0"))):
        print('Args in experiment:')
        print(args)
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii)

        setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


