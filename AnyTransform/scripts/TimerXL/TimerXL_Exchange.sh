export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for seed in 0 1 2 3 4 5 6 7 8 9;do
python -u ./AnyTransform/exp_multi.py \
  --model_names TimerXL \
  --data_names Exchange \
  --pred_lens '24 48 96 192' \
  --num_params 500 \
  --num_samples 500 \
  --ablation none \
  --max_processes 8 \
  --seed $seed
done