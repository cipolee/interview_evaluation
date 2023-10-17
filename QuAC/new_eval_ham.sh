# Run Evaluation (Auto-Rewrite as example)
CUDA_VISIBLE_DEVICES=5 python interview_eval.py \
  --type ham \
  --output_dir outputs/ham_checkpoint \
  --write_dir results/pred \
  --predict_file data/predict.json \
  --qa_log ham.txt \
  --history_ground_truth True \
  --max_seq_length 512 \
  --doc_stride 128 \
  --max_query_length 64 \
  --do_lower_case \
  --history_len 6 \
  --match_metric f1 \
  --add_background \
  --skip_entity \
  --pred \
  --init_checkpoint outputs/ham_checkpoint/model_52000.ckpt \
  --bert_config_file bert-large/bert_config.json \
  --vocab_file bert-large/vocab.txt \
  --MTL_mu 0.8 \
  --MTL_lambda 0.1 \
  --mtl_input reduce_mean \
  --max_answer_length 40 \
  --max_considered_history_turns 4 \
  --bert_hidden 1024 \
  --fine_grained_attention \
  --better_hae \
  --MTL \
  --use_history_answer_marker \
  --start_i \
  --end_i
