CUDA_VISIBLE_DEVICES=5
#python3 -u ./AnyTransform/exp_single.py \
#--res_root_dir new_moti/250203-230542-P500-S500-ABnone-SD0-MP8 \
#--use_gpu \
#--gpu_indexes 0 \
#--data_name ETTh1 \
#--model_name Timer1 \
#--pred_len 24 \
#--num_params 500 \
#--num_samples 500 \
#--ablation none \
#--seed 0

model_name=Timer
seq_len=672
label_len=576
#pred_len=96
#output_len=96
patch_len=96
ckpt_path=/data/qiuyunzhong/LTSM/checkpoints/Timer_forecast_1.0.ckpt

for pred_len in 24;do
python3 -u ./AnyTransform/exp_single.py \
  --task_name forecast \
  --is_training 1 \
  --is_finetuning 1 \
  --seed 1 \
  --ckpt_path  $ckpt_path\
  --root_path ../tslib/dataset/electricity/ \
  --data_path electricity.csv \
  --data_name Electricity \
  --data custom \
  --model_id ECL_full_shot \
  --model Timer \
  --model_name Timer1 \
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