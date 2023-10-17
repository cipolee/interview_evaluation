CUDA_VISIBLE_DEVICES=3 \
python interview_eval.py \
--type bert \
--bert_model bert-base-uncased/ \
--qa_log QA_agent_bert_ground_truth.txt \
--do_predict \
--history_ground_truth True \
--output_dir outputs/bert \
--predict_file data/coqa_400_case.json \
--do_lower_case \
--start_i \
--end_i 


# --history_ground_truth True \

