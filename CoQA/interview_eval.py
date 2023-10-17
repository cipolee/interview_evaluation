# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on CoQA."""

from __future__ import absolute_import, division, print_function
import pdb
import argparse
import logging
import os
import random
import sys
from io import open
import re,string,json
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from modeling import BertForCoQA
from bert import AdamW, WarmupLinearSchedule, BertTokenizer

from run_coqa_dataset_utils import read_coqa_examples_dialog,read_coqa_examples, convert_examples_to_features,convert_one_example_to_features, RawResult, write_predictions, score
# from parallel import DataParallelCriterion, DataParallelModel, gather
from models.graphflow.interface import GraphFlow
from models.graphflow.utils.data_utils import QADataset
from run_coqa_dataset_utils import read_coqa_examples_dialog,read_coqa_examples, convert_examples_to_features,convert_one_example_to_features, RawResult, write_predictions, score
from tqdm import tqdm
from client_work import run_client
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)


def get_graphflow_s_e(example,gold_answer):
    if gold_answer == 'unknown':
        return -1,-1
    else:
        return example['targets'][0],example['targets'][1]

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

    max_f1 = 0
    max_gds = ''
    for ground in gds:

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




def auto_question_generate(prediction,gold_answers,data_idx,args,gold_answers_origin_other,
                           partial_example,model,states,next_unique_id,tokenizer):

    cnt_correct = 0
    max_gd = ''
    state = True

    while prediction!=gold_answers[data_idx]:
        print('in')
        
        #### 取出grounds
        if args.type == 'graphflow':
            grounds = gold_answers_origin_other[data_idx]

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
                else:
                    insert_his = (model.QA_history[-1][0],'Your answer is right, is any another answer?',(gold_answers[data_idx],None,None))
                    model.QA_history.append(insert_his)
            print(f'>>>>matching ground>>>{max_gd}>>>>')
            print(f'>>>>matching f1>>>{max_f1}>>>>')

            break
        #### 退出条件 2. 答案中含有unknown且prediction中有unknown,则结束对话
        elif 'unknown' in gold_answers[data_idx] and 'unknown' in prediction:
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

            new_question = run_client(model.QA_history,' '.join(partial_example['question']['word']),prediction,gold_answers[data_idx])
            partial_example['question']['word'] = new_question.split()
        elif args.type == 'ham':

            new_question = run_client(model.QA_history,partial_example.question_text,prediction,gold_answers[data_idx])
            partial_example.question_text= new_question
        elif args.type == 'excord':

            if '</s></s>' not in partial_example.question_text:
                question = partial_example.question_text
            else:
                question = partial_example.question_text.split('</s></s>')[0]
            new_question = run_client(model.QA_history,question,prediction,gold_answers[data_idx])
            partial_example.question_text= new_question
        else:

            ### bert的格式是
            ### ['what influences does he have in her music?  unknown', 'what collaborations did she do with nikos?  Since 1975, all her releases have become gold or platinum and have included songs by Karvelas.', 'what were some of the songs?']
            new_question = run_client(model.QA_history,partial_example.question_text[-1],prediction,gold_answers[data_idx])
            partial_example.question_text = new_question

                 
        prediction, next_unique_id = model.predict_one_automatic_turn(partial_example,unique_id=next_unique_id, example_idx=data_idx, tokenizer=tokenizer)

        #### 退出条件 4. 在提示情况下模型回答unknown, 此时该答案被转换成ground truth
        if prediction == 'unknown':
            
            state = False
            if args.type == 'graphflow':
                g_s,g_e = get_graphflow_s_e(partial_example,gold_answers[data_idx])
                modify_QA_his = (model.QA_history[-1][0],model.QA_history[-1][1],(gold_answers[data_idx],g_s,g_e))
                model.QA_history[-1]=modify_QA_his
                modify_his = (model.QA_history[-1][1].split(),gold_answers[data_idx].split(),g_s,g_e)
                model.history[-1]=modify_his
            else:
                modify_his = (model.QA_history[-1][0],model.QA_history[-1][1],(gold_answers[data_idx],None,None))
                model.QA_history[-1]=modify_his
            break
        cnt_correct += 1

    states.append(state)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--type",
                        default=None,
                        type=str,
                        required=True,
                        help=".")
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model checkpoints and predictions will be written."
    )

    ## Other parameters
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="CoQA json for training. E.g., coqa-train-v1.0.json")
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="CoQA json for predictions. E.g., coqa-dev-v1.0.json")
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
        "--coqa_answer_class_num",
        default=4,
        type=int,
        help=
        "coqa num answer class")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    # parser.add_argument("--do_F1",
    #                     action='store_true',
    #                     help="Whether to calculating F1 score") # we don't talk anymore. please use official evaluation scripts
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2.0,
                        type=float,
                        help="Total number of training epochs to perform.")
        # HAM specific args
    parser.add_argument('--bert_config_file', type=str, default="SOME_PATH/wwm_uncased_L-24_H-1024_A-16/bert_config.json", help="bert_config.json for bert-large-uncased")
    parser.add_argument('--vocab_file', type=str, default="SOME_PATH/vocab.txt", help="downloadable from https://worksheets.codalab.org/worksheets/0xb92c7222574046eea830c26cd414faec")
    parser.add_argument('--init_checkpoint', type=str, default="SOME_PATH/model_52000.ckpt", help="downloadable from https://worksheets.codalab.org/worksheets/0xb92c7222574046eea830c26cd414faec")
    parser.add_argument('--train_batch_size', type=int, default=16, help="Set to 16 to match training batch size for the original model")
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
    parser.add_argument('--write_dir', type=str)
    parser.add_argument('--match_metric', type=str)
    parser.add_argument('--qa_log', type=str)
    parser.add_argument('--add_background', action='store_true')
    parser.add_argument('--skip_entity', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--out_predictions', action='store_true')
    parser.add_argument('--predict_raw_text', action='store_true')
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
        "--optimizer",
        default='adam',
        help='optimizer')
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
        "--bert_layer_indexes",
        default=[0,12],
        nargs="+",
        help="")
    parser.add_argument(
        "--warmup_proportion",
        default=0.06,
        type=float,
        help=
        "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
        "of training.")
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
    parser.add_argument(
        "--verbose_logging",
        action='store_true',
        help=
        "If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal CoQA evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
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
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help=
        "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help="")
    parser.add_argument(
        '--null_score_diff_threshold',
        type=float,
        default=0.0,
        help=
        "If null_score - best_non_null is greater than the threshold predict null."
    )
    parser.add_argument('--server_ip',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--logfile',
                        type=str,
                        default=None,
                        help='Which file to keep log.')
    parser.add_argument('--logmode',
                        type=str,
                        default=None,
                        help='logging mode, `w` or `a`')
    parser.add_argument('--tensorboard',
                        action='store_true',
                        help='no tensor board')
    parser.add_argument('--qa_tag',
                        action='store_true',
                        help='add qa tag or not')
    parser.add_argument('--start_i',
                        default=0,
                        type=int,
                        help='the first case index')
    parser.add_argument('--end_i',
                        type=int,
                        default=400,
                        help='the last case index')
    parser.add_argument('--history_len',
                        type=int,
                        default=2,
                        help='length of history')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    args = parser.parse_args()
    print(args)

    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.type == 'bert':
        model = BertForCoQA.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
    if args.type == 'graphflow':
        args.model_name_or_path = args.output_dir
        model = GraphFlow(args=args)
        # tokenizer = BertTokenizer.from_pretrained('outputs/bert')
        tokenizer = model.tokenizer()
        with open('data/dev.json','r') as f:
            cases_400=json.load(f)
        eval_examples_g = set([])
        for example in cases_400['data']:
            eval_examples_g.add(example['id'])
    logger.info('loadded model {}'.format(args.type))

    passage_index_list = range(args.start_i,args.end_i)
    if args.type == 'bert':
        eval_examples = read_coqa_examples_dialog(input_file=args.predict_file,
                                               history_len=args.history_len,
                                               add_QA_tag=args.qa_tag)
    else:
        args.dataset_name = 'coqa'
        args.max_answer_len = args.max_answer_length
        eval_examples  = QADataset(args.predict_file, vars(args))
    next_unique_id=1000000000
    f = open(args.qa_log,'a')
    for passage_index in tqdm(passage_index_list):
        para = eval_examples[passage_index]
        if args.type=='bert':
            para_context = para['context']
            para_gold_answers_list = para['gold_answers_list']
            para_gold_answers = para['gold_answers']
            para_examples = para['examples']
        else:
            para_context = para['raw_evidence']
            all_gold_answers_list = [i['answers'] for i in para['turns']]
            para_gold_answers = [i[0] for i in all_gold_answers_list]
            para_gold_answers_list = [i[1:] for i in all_gold_answers_list]
            para_examples = para['turns']
            
        if args.type == 'graphflow' and para['id'] not in eval_examples_g:

            continue
        else:

            pass
        gold_answers_origin_other = [[para_gold_answers[i]]+para_gold_answers_list[i] for i in range(len(para_gold_answers))]
        model.QA_history = []
        if args.type == "graphflow":
            model.history = []
        predictions=[]
        states = []
        record_history = []
        ground_history = []
        ground_history_graphflow = []

        for example_idx,example in enumerate(para_examples):
            # print(example)
            try:
                print(example.qas_id)
            except:
                print(para['id'],example['turn_id'])
            if args.type == 'graphflow':
                example['cid'] = para['id']
                example['raw_evidence'] = para_context
                example['evidence'] = para['evidence']
            #### 记录  ground_history, graphflow需要考虑QA_history和history, ham需要考虑QA_history和QA_history_ham
            if args.type == 'graphflow':
                g_s,g_e = get_graphflow_s_e(example,para_gold_answers[example_idx])
                origin_question = example['question']['word']
                ground_history.append((example_idx+1,' '.join(origin_question),(para_gold_answers[example_idx],g_s,g_e)))
                ground_history_graphflow.append((origin_question,para_gold_answers[example_idx].split(' '),g_s,g_e))
                # modify_QA_his = (model.QA_history[-1][0],model.QA_history[-1][1],(gold_answers[data_idx],g_s,g_e))
            else:
                ground_history.append((example_idx+1,example.question_text,(para_gold_answers[example_idx],None,None)))
            ####
            next_unique_id += 1
            prediction,next_unique_id = model.predict_one_automatic_turn(example,unique_id=next_unique_id,example_idx=example_idx,tokenizer=tokenizer)
            print(prediction,next_unique_id)
            # pdb.set_trace()
            # pdb.set_trace()
            ##### 进入重写bot
            auto_question_generate(prediction,para_gold_answers,example_idx,args,gold_answers_origin_other,
                           example,model,states,next_unique_id,tokenizer)
            #####
            if args.history_ground_truth == 'True':
                if args.type == 'graphflow':

                    record_history.extend([i for i in model.QA_history if i[0]==example_idx+1])
                    model.QA_history = [i for i in ground_history]
                    model.history = [i for i in ground_history_graphflow]
                else:
                    record_history.extend([i for i in model.QA_history if i[0]==example_idx+1])
                    model.QA_history = [i for i in  ground_history]
                    print('hi')
            ### 结束开始下一个人写问题

        if args.history_ground_truth == 'False':
            writen_history = model.QA_history
            if args.type == 'ham':
                writen_history = model.QA_history_ham
        elif args.history_ground_truth == 'True':
            writen_history = record_history
            # record_history = []
        else:
            assert False
        f.write(str(writen_history)+'###'+str(states)+'\n')
        print('@@@@@'+str(writen_history)+'###'+str(states)+'@@@@@'+'\n')
        print(f'>>>>origin turn>>{len(para_examples)}>>>>>modify turn>>>>>>>>')
    f.close()

    # we don't do F1 any more

    # if args.do_F1 and (args.local_rank == -1
    #                    or torch.distributed.get_rank() == 0):
    #     logger.info("Start calculating F1")
    #     cached_eval_examples_file = args.predict_file + '_examples.pk'
    #     try:
    #         with open(cached_eval_examples_file, 'rb') as reader:
    #             eval_examples = pickle.load(reader)
    #     except:
    #         eval_examples = read_coqa_examples(input_file=args.predict_file)
    #     pred_dict = json.load(
    #         open(os.path.join(args.output_dir, "predictions.json"), 'rb'))
    #     truth_dict = {}
    #     for i in range(len(eval_examples)):
    #         answers = eval_examples[i].additional_answers
    #         tmp = eval_examples[i].orig_answer_text
    #         if tmp not in answers:
    #             answers.append(tmp)
    #         truth_dict[eval_examples[i].qas_id] = answers
    #     with open(os.path.join(args.output_dir, "truths.json"), 'w') as writer:
    #         writer.write(json.dumps(truth_dict, indent=4) + '\n')
    #     result, all_f1s = score(pred_dict, truth_dict)
    #     logger.info(str(result))


if __name__ == "__main__":
    main()
