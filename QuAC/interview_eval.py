from __future__ import absolute_import, division, print_function

import argparse
import logging
import random
import sys
from io import open
import pdb
import openai
# import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tqdm import tqdm, trange
import collections
import json,string
import math
import os
import six
import re
import numpy as np
from copy import deepcopy
import itertools
from time import time
import traceback
from collections import Counter
from connection.client_work import run_client
from tqdm import tqdm
# from run_quac_eval_util import rewrite_with_coreference, filter_with_coreference, write_automatic_eval_result, write_invalid_category, load_context_indep_questions
def rewrite_with_coreference(*args):
    pass

def filter_with_coreference(*args):
    pass

def load_context_indep_questions(*args):
    pass

def get_ham_s_e(partial_example,gold_answer):
    s =  ' '.join(partial_example.doc_tokens).index(gold_answer)
    e = s + len(gold_answer) -1
    return s,e
    # return qa_tuples[-1][1],qa_tuples[-1][2]
def write_invalid_category(json_file, skip_dictionary):
    with open(json_file, 'w') as fout:
        fout.write(json.dumps(skip_dictionary, indent=2))

def auto_question_generate(prediction,gold_answers,data_idx,args,gold_answers_origin_other,
                           partial_example,model,states,next_unique_id,tokenizer):

    cnt_correct = 0
    max_gd = ''
    state = True

    while prediction!=gold_answers[data_idx]:
        print('in')
        
        #### 取出grounds
        if args.type == 'graphflow':
            grounds = gold_answers_origin_other

        else:
            grounds = gold_answers_origin_other[data_idx]
        
        max_f1,gd = max_f1_score(prediction,grounds)

        #### 退出条件 1. 如果max_f1比0.5大,则结束对话
        if max_f1>0.5:

            if f1_score(gold_answers[data_idx],gd)<0.5:
                if args.type == 'graphflow':
                    g_s,g_e = get_graphflow_s_e(partial_example,gold_answers[data_idx])
                    insert_his =('Your answer is right, is any another answer?'.split(),gold_answers[data_idx].split(),g_s,g_e)
                    model.history.append(insert_his)
                    insert_QA_his =(model.QA_history[-1][0],'Your answer is right, is any another answer?',(gold_answers[data_idx],g_s,g_e))
                    model.QA_history.append(insert_QA_his)
                elif args.type == 'ham':
                    g_s,g_e = get_ham_s_e(partial_example,gold_answers[data_idx])
                    modify_QA_his = (model.QA_history[-1][0]+1,'Your answer is right, is any another answer?',(gold_answers[data_idx],g_s,g_e))
                    model.QA_history.append(modify_QA_his)


                    modify_QA_his_ham = (model.QA_history_ham[-1][0],'Your answer is right, is any another answer?',(gold_answers[data_idx],g_s,g_e))
                    model.QA_history_ham.append(modify_QA_his_ham)

                else:
                    insert_his = (model.QA_history[-1][0],'Your answer is right, is any another answer?',(gold_answers[data_idx],None,None))
                    model.QA_history.append(insert_his)
            print(f'>>>>matching ground>>>{max_gd}>>>>')
            print(f'>>>>matching f1>>>{max_f1}>>>>')

            break
        #### 退出条件 2. 答案中含有CANNOTANSWER且prediction中有CANNOTANSWER,则结束对话
        elif 'CANNOTANSWER' in gold_answers[data_idx] and 'CANNOTANSWER' in prediction:
            break

        #### 退出条件 3. 进行3次提示还没退出则强制退出
        if cnt_correct >= 3:

            ### 三次都没修正对，则给出正确答案
            prediction = gold_answers[data_idx]
            if args.type == 'graphflow':
                g_s,g_e = get_graphflow_s_e(partial_example,gold_answers[data_idx])
                modify_QA_his = (model.QA_history[-1][0],model.QA_history[-1][1],(gold_answers[data_idx],g_s,g_e))
                model.QA_history[-1]=modify_QA_his
                modify_his = (model.QA_history[-1][1].split(),gold_answers[data_idx].split(),g_s,g_e)
                model.history[-1]=modify_his
            elif args.type == 'ham':
                g_s,g_e = get_ham_s_e(partial_example,gold_answers[data_idx])
                modify_QA_his = (model.QA_history[-1][0],model.QA_history[-1][1],(gold_answers[data_idx],g_s,g_e))
                model.QA_history[-1]=modify_QA_his 


                modify_QA_his_ham = (model.QA_history_ham[-1][0],model.QA_history_ham[-1][1],(gold_answers[data_idx],g_s,g_e))
                model.QA_history_ham[-1]=modify_QA_his_ham
            else:
                modify_his = (model.QA_history[-1][0],model.QA_history[-1][1],(gold_answers[data_idx],None,None))
                model.QA_history[-1]=modify_his
            state = False
            break

        #### 使用chatgpt产生新的问题
        if args.type == 'graphflow':
            # pdb.set_trace()
            new_question = run_client(model.QA_history,' '.join(partial_example['question']['word']),prediction,gold_answers[data_idx])
            partial_example['question']['word'] = new_question.split()
        elif args.type == 'ham':
            # pdb.set_trace()
            new_question = run_client(model.QA_history,partial_example.question_text,prediction,gold_answers[data_idx])
            partial_example.question_text= new_question
        elif args.type == 'excord':
            # pdb.set_trace()
            if '</s></s>' not in partial_example.question_text:
                question = partial_example.question_text
            else:
                question = partial_example.question_text.split('</s></s>')[0]
            new_question = run_client(model.QA_history,question,prediction,gold_answers[data_idx])
            partial_example.question_text= new_question
        else:
            # pdb.set_trace()
            ### bert的格式是
            ### ['what influences does he have in her music?  CANNOTANSWER', 'what collaborations did she do with nikos?  Since 1975, all her releases have become gold or platinum and have included songs by Karvelas.', 'what were some of the songs?']
            new_question = run_client(model.QA_history,partial_example.question_text[-1],prediction,gold_answers[data_idx])
            partial_example.question_text = new_question
        # pdb.set_trace()
                 
        prediction, next_unique_id = model.predict_one_automatic_turn(partial_example,unique_id=next_unique_id, example_idx=data_idx, tokenizer=tokenizer)

        #### 退出条件 4. 在提示情况下模型回答CANNOTANSWER, 此时该答案被转换成ground truth
        if prediction == 'CANNOTANSWER':
            
            state = False
            if args.type == 'graphflow':
                g_s,g_e = get_graphflow_s_e(partial_example,gold_answers[data_idx])
                modify_QA_his = (model.QA_history[-1][0],model.QA_history[-1][1],(gold_answers[data_idx],g_s,g_e))
                model.QA_history[-1]=modify_QA_his
                modify_his = (model.QA_history[-1][1].split(),gold_answers[data_idx].split(),g_s,g_e)
                model.history[-1]=modify_his
            elif args.type == 'ham':
                g_s,g_e = get_ham_s_e(partial_example,gold_answers[data_idx])
                modify_QA_his = (model.QA_history[-1][0],model.QA_history[-1][1],(gold_answers[data_idx],g_s,g_e))
                model.QA_history[-1]=modify_QA_his

                modify_QA_his_ham = (model.QA_history_ham[-1][0],model.QA_history_ham[-1][1],(gold_answers[data_idx],g_s,g_e))
                model.QA_history_ham[-1]=modify_QA_his_ham
            else:
                modify_his = (model.QA_history[-1][0],model.QA_history[-1][1],(gold_answers[data_idx],None,None))
                model.QA_history[-1]=modify_his
            break
        cnt_correct += 1

    states.append(state)

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_graphflow_s_e(example,gold_answer):
    if gold_answer == 'CANNOTANSWER':
        return -1,-1
    else:
        return example['targets'][0],example['targets'][1]

def hack_graphflow_process():
    with open('data/val_400_case.json', "r", encoding="utf-8") as reader:
        input_data = json.load(reader)["data"]

    articles = {}
    for entry in tqdm(input_data):
        for paragraph in entry["paragraphs"]:
            # pdb.set_trace()

            for qa_idx, qa in enumerate(paragraph["qas"]):
                qas_id = qa["id"]
                gold_answers_list=[i['text'] for i in qa['answers']]


                a = {qas_id:gold_answers_list}
                articles.update(a)
    return articles

def f1_score(prediction, ground_truth):
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def max_f1_score(prediction,gds):
    # pdb.set_trace()
    max_f1 = 0
    max_gds = ''
    for ground in gds:
        # pdb.set_trace()
        f1 = 0
        pred_len = len(prediction.split())
        gd_len = len(ground.split())
        for i in [0,1,2]:
            for j in [0,1,2]:

                if i==0:
                    pd=prediction
                elif i==1:
                    pd=' '.join(prediction.split()[:pred_len//2])
                else:
                    pd=' '.join(prediction.split()[pred_len//2:])
                if j==0:
                    gd=ground
                elif j==1:
                    gd=' '.join(ground.split()[:gd_len//2])
                else:
                    gd=' '.join(ground.split()[gd_len//2:])        
                f1 = f1_score(pd,gd)
                if max_f1<f1:
                    max_f1 = f1
                    max_gds = ground
    print(f'>>>>pd>>>>>>>>>>>>>{prediction}')
    print(f'>>>>gd>>>>>>>>>>>>>{max_gds}')
    print(f'<<<<max_f1<<<<<<<<<<{max_f1}')
    return max_f1,max_gds




def get_response(prompt, temperature=0.1, max_tokens=2048):
    """调用函数，max_tokens是要生成的tokens的最大长度， 
        max_tokens+ num_prompt_tokens最多是4096 受调用的model限制"""
    response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.3,
    )
    return response

def question_rewrite(history,current_question,ground_truth,pred_answer):
    conversational_history  = ""
    ## 声明任务
    ## 给出对话历史
    ## 给出现状
    ## 请求给出当前问题
    prompt = f"""
    This is a turing test. You play a teacher and someone else plays a student, you ask a question, and if you judge the student to answer correctly, you keep asking questions. Otherwise you need to prompt the student for the wrong answer and give the student some hints based on the correct answer without revealing the answer. The prompt is a single-hop question and end the conversation, and should not be redundant.

    this is history conversation
    {conversational_history}

    Then when you ask '{current_question}', the student answer '{pred_answer}', however the gold answer is 'ground_truth'
    What question do you ask to prompt students? The question is a single-hop and short!
    """
    
    return prompt


def write_automatic_eval_result(json_file, evaluation_result):
    """evaluation_results = [{"CID": ..., 
                              "Predictions": [
                                  (qa_id, span),
                                  ...
                              ]}, ...]"""

    with open(json_file, 'w') as fout:
        for passage_index, predictions in evaluation_result.items():
            output_dict = {'best_span_str': [], 'qid': [], 'yesno':[], 'followup': []}
            for qa_id, span in predictions["Predictions"]:
                output_dict['best_span_str'].append(span)
                output_dict['qid'].append(qa_id)
                output_dict['yesno'].append('y')
                output_dict['followup'].append('y')
            fout.write(json.dumps(output_dict) + '\n')

logger = logging.getLogger(__name__)

def main():
    print("Program start")     
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--type",
                        default='bert',
                        type=str,
                        choices=['graphflow', 'bert', 'ham', 'excord'],
                        required=True,
                        help="Aliases for model to evaluate. Eg. 'bert', 'ham', 'excord', 'graphflow.")
    parser.add_argument(
        "--output_dir",
        default=None,
        required=True,
        type=str,
        help="The output directory where the model checkpoints to be evaluated is written."
    )
    parser.add_argument(
        "--write_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where evaluation results will be stored."
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        required=True,
        help="Path to QuAC dev file. Download from: https://s3.amazonaws.com/my89public/quac/val_v0.2.json")
    
    # Eval specific parameters
    parser.add_argument('--history_len',
                        type=int,
                        default=2,
                        help='length of history')
    parser.add_argument('--start_i', type=int, default=0, help="start passage index of evaluation")
    parser.add_argument('--end_i', type=int,
                        default=1000, help="end passage index of evaluation")
    parser.add_argument('--match_metric', type=str, default='f1', choices=['f1', 'em'], help="which metric to use for detecting invalid questions")
    parser.add_argument('--add_background', action='store_true', help="Whether or not to add background section during validity evaluation")
    parser.add_argument('--skip_entity', action='store_true', help="Whether to ignore special entities (e.g. named entity) in validity evaluation")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Turn on if using Auto-Replace by replacing the original question with context-independent questions"
    )
    parser.add_argument(
        "--canard_path",
        default="/n/fs/nlp-data/conversational-qa/canard/test.json",
        type=str,
        help="The path to CANARD test set, which is QuAC dev set with context-independent questions."
    )
    parser.add_argument(
        "--qa_log",
        default="",
        type=str,
        help="The path to CANARD test set, which is QuAC dev set with context-independent questions."
    )
    
    parser.add_argument(
        '--rewrite',
        action="store_true",
        help="Turn on if using Auto-Rewrite by rewriting the question with coreference model"
    )
    parser.add_argument(
        '--pred',
        action="store_true",
        help="Turn on if using Auto-Pred by only using predicted answer as history without replacing or rewriting invalid questions."
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help=
        "When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help=
        "The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help=
        "The total number of n-best predictions to generate in the nbest_predictions.json "
        "output file.")
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help=
        "The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help=
        "Whether to lower case the input text. True for uncased models, False for cased models."
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--fp16',
        action='store_true',
        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help=
        "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument(
        '--null_score_diff_threshold',
        type=float,
        default=0.0,
        help=
        "If null_score - best_non_null is greater than the threshold predict null."
    )
    parser.add_argument('--logfile',
                        type=str,
                        default=None,
                        help='Which file to keep log.')
    parser.add_argument('--logmode',
                        type=str,
                        default=None,
                        help='logging mode, `w` or `a`')
    parser.add_argument(
        "--verbose_logging",
        action='store_true',
        help=
        "If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal QuAC evaluation.")
    
    # Bert specific
    parser.add_argument('--rationale_beta', type=float, default=0,
                        help="Multiplier for rationale loss.")

    # HAM specific args
    parser.add_argument('--bert_config_file', type=str, default="SOME_PATH/wwm_uncased_L-24_H-1024_A-16/bert_config.json", help="bert_config.json for bert-large-uncased")
    parser.add_argument('--vocab_file', type=str, default="SOME_PATH/vocab.txt", help="downloadable from https://worksheets.codalab.org/worksheets/0xb92c7222574046eea830c26cd414faec")
    parser.add_argument('--init_checkpoint', type=str, default="SOME_PATH/model_52000.ckpt", help="downloadable from https://worksheets.codalab.org/worksheets/0xb92c7222574046eea830c26cd414faec")
    parser.add_argument('--train_batch_size', type=int, default=16, help="Set to 16 to match training batch size for the original model")
    parser.add_argument('--predict_batch_size', type=int, default=16, help="Set to 16 to match training batch size for the original model")
    parser.add_argument('--max_history_turns', type=int, default=11)
    parser.add_argument('--max_considered_history_turns', type=int, default=4)
    parser.add_argument('--MTL_lambda', type=float, default=0.1)
    parser.add_argument('--MTL_mu', type=float, default=0.8)
    parser.add_argument('--history_selection', type=str, default="previous_j")
    parser.add_argument('--history_attention_input', type=str, default="reduce_mean")
    parser.add_argument('--mtl_input', type=str, default="reduce_mean")
    parser.add_argument('--history_ngram', type=int, default=1)
    parser.add_argument('--bert_hidden', type=int, default=1024)
    parser.add_argument('--only_history_answer', action='store_true')
    parser.add_argument('--use_history_answer_marker', action='store_true')
    parser.add_argument('--better_hae', action='store_true')
    parser.add_argument('--MTL', action='store_true')
    parser.add_argument('--disable_attention', action='store_true')
    parser.add_argument('--history_attention_hidden', action='store_true')
    parser.add_argument('--reformulate_question', action='store_true')
    parser.add_argument('--front_padding', action='store_true')
    parser.add_argument('--fine_grained_attention', action='store_true')
    parser.add_argument('--append_self', action='store_true')

    # GraphFlow specific args
    parser.add_argument(
        "--embed_file",
        default='/n/fs/nlp-huihanl/conversational-qa/local/Bert4QuAC/glovecove/glove.840B.300d.txt',
        type=str,
        help="GloVE embedding file. Downloadable from glovecove.")
    parser.add_argument(
        "--saved_vocab_file",
        default='/n/fs/nlp-huihanl/conversational-qa/GraphFlow/data/quac/word_model_min_5',
        type=str,
        help="Saved vocab file after training.")
    parser.add_argument(
        "--pretrained",
        default='/n/fs/nlp-huihanl/conversational-qa/GraphFlow/out/quac/graphflow_dynamic_graph',
        type=str,
        help="Saved model after training.")
    
    # Processing data
    parser.add_argument(
        "--min_freq",
        default=5,
        type=int,
        help="")
    parser.add_argument(
        "--top_vocab",
        default=200000,
        type=int,
        help="")
    parser.add_argument(
        "--n_history",
        default=2,
        type=int,
        help="")
    parser.add_argument(
        "--max_turn_num",
        default=20,
        type=int,
        help="")
    parser.add_argument(
        "--no_pre_question",
        action="store_true",
        help="")
    parser.add_argument(
        "--no_pre_answer",
        action="store_true",
        help="")
    parser.add_argument(
        "--embed_type",
        default='glove',
        type=str,
        help="")
    parser.add_argument(
        "--vocab_embed_size",
        default=300,
        type=int,
        help="")
    parser.add_argument(
        "--fix_vocab_embed",
        action="store_true",
        help="")
    parser.add_argument(
        "--f_qem",
        action="store_true",
        help="")
    parser.add_argument(
        "--f_pos",
        action="store_true",
        help="")
    parser.add_argument(
        "--f_ner",
        action="store_true",
        help="")
    parser.add_argument(
        "--f_tf",
        action="store_true",
        help="")
    parser.add_argument(
        "--use_ques_marker",
        action="store_true",
        help="")
    parser.add_argument(
        "--ctx_exact_match_embed_dim",
        default=3,
        type=int,
        help="")
    parser.add_argument(
        "--ctx_pos_embed_dim",
        default=12,
        type=int,
        help="")
    parser.add_argument(
        "--ctx_ner_embed_dim",
        default=8,
        type=int,
        help="")
    parser.add_argument(
        "--answer_marker_embed_dim",
        default=10,
        type=int,
        help="")
    parser.add_argument(
        "--ques_marker_embed_dim",
        default=3,
        type=int,
        help="")
    parser.add_argument(
        "--ques_turn_marker_embed_dim",
        default=5,
        type=int,
        help="")
    parser.add_argument(
        "--hidden_size",
        default=300,
        type=int,
        help="")
    parser.add_argument(
        "--word_dropout",
        default=0.3,
        type=float,
        help="")
    parser.add_argument(
        "--bert_dropout",
        default=0.4,
        type=float,
        help="")
    parser.add_argument(
        "--rnn_dropout",
        default=0.3,
        type=float,
        help="")
    parser.add_argument(
        "--use_gnn",
        action="store_true",
        help="")
    parser.add_argument(
        "--bignn",
        action="store_true",
        help="")
    parser.add_argument(
        "--static_graph",
        action="store_true",
        help="")
    parser.add_argument(
        "--temporal_gnn",
        action="store_true",
        help="")
    parser.add_argument(
        "--ctx_graph_hops",
        default=3,
        type=int,
        help="")
    parser.add_argument(
        "--ctx_graph_topk",
        default=10,
        type=int,
        help="")
    parser.add_argument(
        "--graph_learner_num_pers",
        default=1,
        type=int,
        help="")
    parser.add_argument(
        "--use_spatial_kernels",
        action="store_true",
        help="")
    parser.add_argument(
        "--use_position_enc",
        action="store_true",
        help="")
    parser.add_argument(
        "--n_spatial_kernels",
        default=3,
        type=int,
        help="")
    parser.add_argument(
        "--max_position_distance",
        default=160,
        type=int,
        help="")
    parser.add_argument(
        "--position_emb_size",
        default=50,
        type=int,
        help="")
    parser.add_argument(
        "--use_bert",
        action="store_true",
        help="")
    parser.add_argument(
        "--finetune_bert",
        action="store_true",
        help="")
    parser.add_argument(
        "--use_bert_weight",
        action="store_true",
        help="")
    parser.add_argument(
        "--history_ground_truth",
        default='False',
        type=str)
    parser.add_argument(
        "--use_bert_gamma",
        action="store_true",
        help="")
    parser.add_argument(
        "--bert_max_seq_len",
        default=500,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--bert_doc_stride",
        default=128,
        type=int,
        help=
        "When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        "--bert_dim",
        default=1024,
        type=int,
        help="")
    parser.add_argument(
        "--bert_model",
        default='bert-large-uncased',
        type=str,
        help="")
    parser.add_argument(
        "--bert_layer_indexes",
        default=[0,12],
        nargs="+",
        help="")
    parser.add_argument(
        "--optimizer",
        default='adamax',
        type=str,
        help="")
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="")
    parser.add_argument(
        "--grad_clipping",
        default=10,
        type=int,
        help="")
    parser.add_argument(
        "--max_answer_len",
        default=35,
        type=int,
        help=
        "The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.")
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help=
        "")
    parser.add_argument(
        "--grad_accumulated_steps",
        default=1,
        type=int,
        help=
        "")
    parser.add_argument(
        "--test_batch_size",
        default=1,
        type=int,
        help=
        "")
    parser.add_argument(
        "--max_epochs",
        default=1000,
        type=int,
        help=
        "")
    parser.add_argument(
        "--patience",
        default=10,
        type=int,
        help=
        "")
    parser.add_argument(
        "--verbose",
        default=1000,
        type=int,
        help=
        "")
    parser.add_argument(
        "--unk_answer_threshold",
        type=float,
        default=0.3,
        help="",
    )
    parser.add_argument(
        "--out_predictions", action="store_true", help=""
    )
    parser.add_argument(
        "--predict_raw_text", action="store_true", help=""
    )
    parser.add_argument(
        "--out_pred_in_folder", action="store_true", help=""
    )
    parser.add_argument(
        "--shuffle", action="store_true", help=""
    )
    parser.add_argument(
        "--cuda_id",
        default=0,
        type=int,
        help=
        "")

    args = parser.parse_args()

    # Set model_class. You can add your own model here.
    if args.type == "bert":
        from models.bert.interface import BertOrg
        model_class = BertOrg
        if not args.do_lower_case:
            logger.warn("You probably want to use --do_lower_case when using BERT.")
    elif args.type == "ham":
        from models.ham.interface import BertHAM
        model_class = BertHAM
        if not args.do_lower_case:
            logger.warn("You probably want to use --do_lower_case when using BERT.")
        import tensorflow as tf
        tf.set_random_seed(args.seed)
        tf.logging.set_verbosity(tf.logging.INFO)
        device = None
    elif args.type == "excord":
        from models.excord.interface import Excord
        model_class = Excord
        if args.do_lower_case:
            logger.warn("Do not use --do_lower_case when using Excord.")
        import torch
        torch.manual_seed(args.seed)
    elif args.type == "graphflow":
        from models.graphflow.interface import GraphFlow
        model_class = GraphFlow
        if args.do_lower_case:
            logger.warn(
                "Do not use --do_lower_case when using GraphFlow")
    else:
        raise NotImplementedError
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        filename=args.logfile,
        filemode=args.logmode)
    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.predict_file:
        raise ValueError(
            "If `do_predict` is True, then `predict_file` must be specified."
        )

    if not os.path.exists(args.output_dir):
        raise ValueError(
            "Output directory {} does not contain model checkpoint.".format(args.output_dir))
    
    args.model_name_or_path=args.output_dir
    logger.info("***** Loading pre-trained model *****")
    logger.info("Model directory: %s", args.output_dir)
    
    # initialize model and tokenizer
    model = model_class(args=args)
    tokenizer = model.tokenizer()

    # load predict file
    partial_eval_examples_file = args.predict_file
    ###### modify
    # load partially filled examples according to your model
    partial_examples = model.load_partial_examples(partial_eval_examples_file)
    

    # Set list of QuAC passages to evaluate on
    passage_index_list = range(args.start_i,args.end_i)

    next_unique_id=1000000000
    f = open(args.qa_log,'a')
    if args.type == 'graphflow':
        gold_answers_list_dict = hack_graphflow_process()
        # pdb.set_trace()
    cnt = 0
    Flag_Graphflow = False
    for passage_index in passage_index_list:
        
        
        passage_index = int(passage_index)

        # read into a list of paragraph examples
        paragraph = partial_examples[passage_index] 
        examples = paragraph["examples"]
        gold_answers = paragraph["gold_answers"] # a list of gold answers
        # pdb.set_trace()
        if args.type == 'graphflow':
            if examples[0]['turn_id'] not in gold_answers_list_dict:

                print('hi')
                continue
        else:
            gold_answers_list = paragraph["gold_answers_list"]
            try:
                gold_answers_origin_other = [[gold_answers[i]]+gold_answers_list[i] for i in range(len(gold_answers))]
            except Exception as e:
                print(e)
                pdb.set_trace()
        


        # clear model QA history for new passage
        model.QA_history = []


        ground_history = []
        record_history = []


        ground_history_graphflow = []
        ground_history_ham = []


        
        # A quick hack for graphflow
        if args.type == "graphflow":
            model.history = []
        if args.type == 'ham':
            model.QA_history_ham = []

        # A quick hack for graphflow
        predictions=[]
        states = []
        cnt += 1
        print(cnt)
        for data_idx in range(len(examples)):
            partial_example = examples[data_idx]

            # A quick hack for graphflow, 获得graphflow的ground truth
            if args.type == "graphflow":
                ## hack 2023.6.3

                

                gold_answers_origin_other = [gold_answers[data_idx]]+gold_answers_list_dict[partial_example['turn_id']]
                attr_partial_example = model.convert_example(partial_example)
            else:
                attr_partial_example = partial_example


            #### 记录  ground_history, graphflow需要考虑QA_history和history, ham需要考虑QA_history和QA_history_ham
            if args.type == 'graphflow':
                g_s,g_e = get_graphflow_s_e(partial_example,gold_answers[data_idx])
                origin_question = partial_example['question']['word']
                ground_history.append((data_idx,' '.join(origin_question),(gold_answers[data_idx],g_s,g_e)))
                ground_history_graphflow.append((origin_question,gold_answers[data_idx].split(' '),g_s,g_e))
                # modify_QA_his = (model.QA_history[-1][0],model.QA_history[-1][1],(gold_answers[data_idx],g_s,g_e))
            elif args.type == 'ham':
                g_s,g_e  = get_ham_s_e(partial_example,gold_answers[data_idx])
                ground_history.append((data_idx, partial_example.question_text, (gold_answers[data_idx], g_s, g_e)))
                ground_history_ham.append((data_idx, partial_example.question_text, (gold_answers[data_idx], g_s, g_e)))
            else:
                ground_history.append((data_idx,partial_example.question_text,(gold_answers[data_idx],None,None)))
            ####


            logger.info("Valid {}".format(attr_partial_example.qas_id))
            prediction, next_unique_id = model.predict_one_automatic_turn(partial_example,unique_id=next_unique_id, example_idx=data_idx, tokenizer=tokenizer)
            pdb.set_trace()
            predictions.append((attr_partial_example.qas_id, prediction))



            ##### 进入重写bot
            auto_question_generate(prediction,gold_answers,data_idx,args,gold_answers_origin_other,
                           partial_example,model,states,next_unique_id,tokenizer)
            #####


            if args.history_ground_truth == 'True':
                if args.type == 'graphflow':
                    record_history.extend([i for i in model.QA_history if i[0]==data_idx])
                    model.QA_history = [i for i in ground_history]
                    model.history = [i for i in ground_history_graphflow]
                elif args.type == 'ham':
                    record_history.extend([i for i in model.QA_history_ham if i[0]==data_idx])
                    model.QA_history = [i for i in ground_history]
                    model.QA_history_ham = [i for i in ground_history_ham]
                else:
                    record_history.extend([i for i in model.QA_history if i[0]==data_idx])
                    model.QA_history = [i for i in  ground_history]
                    print('hi')
            ### 结束开始下一个人写问题
        
        if args.history_ground_truth == 'False':
            writen_history = model.QA_history
            if args.type == 'ham':
                # pdb.set_trace()
                writen_history = model.QA_history_ham
        elif args.history_ground_truth == 'True':
            writen_history = record_history
            # record_history = []
        else:
            assert False
        f.write(str(writen_history)+'###'+str(states)+'\n')
        logger.info('@@@@@'+str(writen_history)+'###'+str(states)+'@@@@@'+'\n')
        print(f'>>>>origin turn>>{len(examples)}>>>>>modify turn>>>>>>>>')

    f.close()


if __name__ == "__main__":
    main()
    # prompt = '1+1='
    # ans = get_response(prompt, temperature=0.1, max_tokens=2048)
    # pdb.set_trace()
 
"""

case0 : >>>>origin turn>>8>>>>>modify turn>>13>>>>>>
case1 : >>>>origin turn>>8>>>>>modify turn>>14>>>>>>
"""