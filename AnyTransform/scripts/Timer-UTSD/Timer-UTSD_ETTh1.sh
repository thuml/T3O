export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u ./AnyTransform/exp_multi.py \
  --model_names Timer-UTSD \
  --data_names ETTh1 \
  --pred_lens '24 48 96 192' \
  --num_params 500 \
  --num_samples 500 \
  --ablation none \
  --max_processes 8 \
  --seed 0