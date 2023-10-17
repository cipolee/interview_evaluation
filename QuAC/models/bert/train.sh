# Run Training
# important for BERT
directory_to_save_model="../../outputs"
path_to_quac_train_file="../../data/train_v0.2.json"
echo ${path_to_quac_train_file}
CUDA_VISIBLE_DEVICES=7 python3 run_quac_train.py \
  --type bert \
  --model_name_or_path ../../bert-base-uncased \
  --do_train \
  --output_dir ${directory_to_save_model} \
  --overwrite_output_dir \
  --train_file ${path_to_quac_train_file} \
  --train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 512 \
  --learning_rate 3e-5 \
  --history_len 2 \
  --warmup_proportion 0.1 \
  --max_grad_norm -1 \
  --weight_decay 0.01 \
  --rationale_beta 0 \
  --do_lower_case