#export CUDA_VISIBLE_DEVICES=5

model_name=Timer-UTSD
seq_len=672
label_len=576
#pred_len=96
#output_len=96
patch_len=96
ckpt_path=/data/qiuyunzhong/ts_adaptive_inference/Timer/ckpt/Building_timegpt_d1024_l8_p96_n64_new_full.ckpt

for pred_len in 24 48 96 192;do
python3 -u ./AnyTransform/exp_single.py \
  --task_name forecast \
  --is_training 1 \
  --is_finetuning 1 \
  --seed 1 \
  --ckpt_path $ckpt_path\
  --root_path ../DATA/exchange_rate/ \
  --data_path exchange_rate.csv \
  --data_name exchange_rate \
  --data custom \
  --model_id exchange_rate_postHybrid \
  --model $model_name \
  --model_name $model_name \
  --features S \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $pred_len \
  --e_layers 8 \
  --factor 3 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 1024 \
  --learning_rate 3e-5 \
  --num_workers 4 \
  --patch_len $patch_len \
  --train_test 0 \
  --itr 1 \
  --gpu 0 \
  --finetune_epochs 20 \
  --num_params 500 \
  --num_samples 500 \
  --ablation none
done