import argparse
from argparse import Namespace
import os
import datetime

date_time_str = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
res_root_dir = os.path.join('new_moti', date_time_str)

parser = argparse.ArgumentParser(description='Hyperparameter tuning for time-series forecasting')
# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[forecast, imputation, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Timer',
                    help='model name, options: [Timer TrmEncoder]')
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
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--subset', type=int, default=None,
                    help='number of subset samples (use the first ${subset} samples in the dataloader)')

# model define
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
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
parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training',
                    default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

parser.add_argument('--stride', type=int, default=1, help='stride')
parser.add_argument('--ckpt_path', type=str, default='', help='ckpt file')
parser.add_argument('--finetune_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--finetune_rate', type=float, default=0.1, help='finetune ratio')
parser.add_argument('--local_rank', type=int, default=0, help='local_rank')

parser.add_argument('--patch_len', type=int, default=24, help='input sequence length')
parser.add_argument('--subset_rand_ratio', type=float, default=1, help='mask ratio')
parser.add_argument('--data_type', type=str, default='custom', help='data_type')

parser.add_argument('--decay_fac', type=float, default=0.75)

# parameter freeze
parser.add_argument('--freeze_decoder', action='store_true', help='freeze the decoder layer in fintuning',
                    default=False)

# cosin decay
parser.add_argument('--cos_warm_up_steps', type=int, default=100)
parser.add_argument('--cos_max_decay_steps', type=int, default=60000)
parser.add_argument('--cos_max_decay_epoch', type=int, default=10)
parser.add_argument('--cos_max', type=float, default=1e-4)
parser.add_argument('--cos_min', type=float, default=2e-6)

# weight decay
parser.add_argument('--use_weight_decay', type=int, default=0, help='use_post_data')
parser.add_argument('--weight_decay', type=float, default=0.01)

# autoregressive configs
parser.add_argument('--use_ims', action='store_true', help='Iterated multi-step', default=False)
parser.add_argument('--output_len', type=int, default=96, help='output len')
parser.add_argument('--output_len_list', type=int, nargs="+", help="output_len_list")

# train_test
parser.add_argument('--train_test', type=int, default=1, help='train_test')
parser.add_argument('--is_finetuning', type=int, default=1, help='status')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# imputation task & anomaly detection task
parser.add_argument('--use_mask', action='store_true',
                    help='apply masking to input data in auto-encoding form anomaly detection', default=False)
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
parser.add_argument('--mask_type', type=str, default=None, help='mask type')
parser.add_argument('--use_ensemble_forecast', action='store_true',
                    help='use ensemble forecasting in auto-regressive form anomaly detection', default=False)
parser.add_argument('--ensemble_type', type=str, default=None, help='ensemble type')

# visualization
parser.add_argument('--show_embedding', action='store_true', help='plot embedding tsne result', default=False)
parser.add_argument('--show_feature', action='store_true', help='plot feature tsne result', default=False)
parser.add_argument('--show_score', action='store_true', help='plot score tsne result', default=False)
parser.add_argument('--date_record', action='store_true', help='record date in visualization', default=False)

# tsne setting
parser.add_argument('--tsne_perplexity', type=int, default=10, help='The number of neighbor points considered in TSNE algorithm, normally 5 - 50. \
                                        Bigger perplexity leads to less detailed characteristics and smaller perplexity leads to overfit.')
parser.add_argument('--use_PCA', action='store_true',
                    help='using PCA can reduce overall dimensionality and reduce computation resource assumption',
                    default=False)

# T3O Original Params
parser.add_argument('--data_name', type=str, help='Name of the dataset')
parser.add_argument('--model_name', type=str, help='Name of the model')
# parser.add_argument('--pred_len', type=int, required=True, help='Length of prediction')
parser.add_argument('--res_root_dir', type=str, default=f'{res_root_dir}', help='Directory to save results')
# parser.add_argument('--use_gpu', action='store_true', help='Whether to use GPU')
parser.add_argument('--gpu_indexes', type=str, default=None, help='Indexes of GPUs to use')
# override fastmode
parser.add_argument('--fast_mode', action='store_true', help='Whether to use fast mode')
# seed
# parser.add_argument('--seed', type=int, default=0, help='Random seed')  # FIXME: 0
# num_params num_samples ablation
parser.add_argument('--num_params', type=int, default=0, help='Number of parameters to try')
parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to try')
parser.add_argument('--ablation', type=str, default='none', help='Ablation study type')
parser.add_argument('--max_processes', type=int, default=1, help='Maximum number of processes to use')

args = parser.parse_args()

# FIXME:
# if args.fast_mode:
#     args = argparse.Namespace(data_name='ETTh1', model_name='Chronos-tiny', pred_len=720, res_root_dir='./debug',
#                               use_gpu=True, gpu_indexes='0', fast_mode=True)
# python3 ./AnyTransform/exp_single.py --data_name ETTh1 --model_name Chronos-tiny --pred_len 720 --res_root_dir ./debug --use_gpu --gpu_indexes 0 --fast_mode
# python3 ./AnyTransform/exp_single.py --data_name ETTh1 --model_name Chronos-tiny --pred_len 720 --res_root_dir ./debug --fast_mode
# CUDA_VISIBLE_DEVICES='1' python3 ./AnyTransform/exp_single.py --res_root_dir ./debug --use_gpu --gpu_indexes 1 --data_name Electricity --model_name Timer-UTSD --pred_len 96 >./debug/exp.log
# CUDA_VISIBLE_DEVICES='1' python3 ./AnyTransform/exp_single.py --res_root_dir ./debug --use_gpu --gpu_indexes 1 --data_name Electricity --model_name Timer-UTSD --pred_len 96  --num_params 25 --num_samples 100 --ablation none --seed 3 >./debug/exp.log
# 我希望他在判断os是mac的时候fast_mode=True
# fast_mode = True if sys.platform == 'darwin' else False

model_name = args.model_name
data_name = args.data_name
pred_len = args.pred_len
use_gpu = args.use_gpu
gpu_indexes = args.gpu_indexes
res_root_dir = args.res_root_dir
fast_mode = args.fast_mode
seed = args.seed
num_params = args.num_params
num_samples = args.num_samples
ablation = args.ablation
max_processes = args.max_processes

# patch_len = 96
# nan_inf_clip_factor = 5
