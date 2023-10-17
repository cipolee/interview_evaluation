CUDA_VISIBLE_DEVICES=3 python3 interview_eval.py \
--type graphflow \
--output_dir outputs/graphflow/ \
--write_dir results/pred \
--predict_file data/dev.json \
--embed_file outputs/graphflow/glove.840B.300d.txt \
--pretrained outputs/graphflow \
--saved_vocab_file data/word_model_min_5 \
--match_metric f1 \
--qa_log QA_agent_graphflow.txt \
--add_background \
--skip_entity \
--fix_vocab_embed \
--f_qem  \
--f_pos  \
--f_ner  \
--use_ques_marker \
--use_gnn \
--temporal_gnn \
--use_bert \
--use_bert_weight \
--shuffle \
--history_ground_truth True \
--out_predictions \
--predict_raw_text \
--bert_doc_stride 250 \
--bert_model /home/xbli/EvalConvQA-CoQA/outputs/bert \
--bert_dim 768 \
--start_i  \
--end_i 

#
# --qa_log QA_agent_graphflow.txt \
# --history_ground_truth True \