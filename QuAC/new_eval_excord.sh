CUDA_VISIBLE_DEVICES=5 python3 interview_eval.py \
--type excord \
--output_dir outputs/excord_korea \
--write_dir results/pred \
--predict_file data/predict.json \
--match_metric f1 \
--qa_log excord.txt \
--history_ground_truth True \
--add_background \
--skip_entity \
--pred \
--start_i \
--end_i 
