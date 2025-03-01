#!/bin/sh

export CUDA_VISIBLE_DEVICES=4
#export CUDA_LAUNCH_BLOCKING=1

model_name=Timer
seq_len=672
label_len=576
pred_len=96
output_len=96
patch_len=96

for learning_rate in 3e-5
do
for data in ETTh1
do
for subset_rand_ratio in 1
do
i=1
for ckpt_path in ckpt/Building_timegpt_d1024_l8_new_full.ckpt
do
# train
# 5效5闪到调配槽后循环母液Nk,5效5闪到调配槽后循环母液ak/Rp,6效出料密度,5闪出料密度,57效循环母液进料温度,57效循环母液进料密度,冷凝器出汽压力,5效5闪蒸发母液去循环母液调配槽总流量,2效蒸发器汽温,2效蒸发器汽室压力,2效原液换热前温度,3效蒸发器汽温,3效蒸发器汽室压力,3效原液换热前温度,4效蒸发器汽温,4效蒸发器汽室压力,4效原液换热前温度,5效蒸发器汽温,5效蒸发器汽室压力,5效原液换热前温度,5效循环母液进料流量,6效蒸发器汽温,6效蒸发器汽室压力,6效原液换热前温度,原液闪蒸器循环母液进料流量,原液闪蒸器出料温度,原液闪蒸器蒸汽压力,7效原液换热前温度,7效蒸发器汽温,7效蒸发器汽室压力,1闪出料温度,2闪出料温度,3闪出料温度,4闪出料温度,5闪出料温度,5闪去循环母液调配槽密度,5闪去循环母液调配槽流量,新蒸汽进料压力,新蒸汽进料流量,新蒸汽进料温度,循环上水温度,循环上水流量,循环下水温度,原液闪蒸器循环母液进料阀开度,1效出料阀门开度,1闪出料阀开度,3闪出料阀开度,4闪出料阀开度,2效到1效泵开度,4效到3效泵开度,5效到4效泵开度,6效出料出料泵开度,1效蒸发器汽室压力,1效原液换热前温度,1效出料温度,1效蒸发器汽温 \
python -u ./Timer/run.py \
  --task_name large_finetune \
  --is_training 0 \
  --seed 1 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --data $data \
  --model_id 2G_{$seq_len}_{$pred_len}_{$patch_len}_ \
  --model $model_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --e_layers 8 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 32 \
  --finetune_epochs 50 \
  --learning_rate $learning_rate \
  --num_workers 4 \
  --patch_len $patch_len \
  --train_test 1 \
  --subset_rand_ratio $subset_rand_ratio \
  --train_offset $patch_len \
  --itr 1 \
  --gpu 0 \
  --roll \
  --show_demo \
  --is_finetuning 0
i=$((i+2))
done
done
done
done