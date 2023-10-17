import os
# import pandas as pd
# import numpy as np
import re
import string
from collections import defaultdict,Counter
import pdb
## 平均成功轮数
## 不成功比例
## 轮数占比

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
def parse_txt(txt_path,answerble):
    success_dict = defaultdict(int)
    failed_dict = defaultdict(int)
    success_dict_unanswerble = defaultdict(int)
    failed_dict_unanswerble = defaultdict(int)
    cnt_convert = 0
    
    with open(txt_path,'r') as f:

        data = f.read().split('\n')
    data.pop()
    # pdb.set_trace()
    answer_one = 0
    s_cnt,f_cnt = 0,0
    for i in data:
        try:
            history_str,states_str = i.split('###')
            history = eval(history_str)
            states = eval(states_str)
            cnt_list = [0]*len(states)
            turn_answers = [[(0,'')]]
            for j in history:
                if 'Your answer is right' in j[1]:
                    # print('hi')
                    answer_one += 1
                    continue
                try:
                    cnt_list[int(j[0])]+=1
                except:
                    pdb.set_trace()
                finally:
                    if int(j[0]) == turn_answers[-1][0][0]:
                        turn_answers[-1].append((int(j[0]),j[2][0]))
                    else:
                        turn_answers.append([])
                        turn_answers[-1].append((int(j[0]),j[2][0]))
            for j in range(len(states)):
                # if j==0:
                #     qas= turn_answers[j][1:]
                # else:
                #     qas= turn_answers[j]
                # for qa in qas[:-1]:
                #     # pdb.set_trace()
                #     if f1_score(qa[1],qas[-1][1])>0.3:
                #         pdb.set_trace()
                # continue
                # print(len(turn_answers),j)
                if states[j]:
                    success_dict[cnt_list[j]] += 1
                    if 'CANNOTANSWER' in turn_answers[j][-1][1]:
                        success_dict_unanswerble[cnt_list[j]] += 1
                    if 'CANNOTANSWER' in turn_answers[j][0][1] and 'CANNOTANSWER' not in turn_answers[j][-1][1]:
                        s_cnt += 1
                else:
                    if 'CANNOTANSWER' in turn_answers[j][0][1]:
                        f_cnt += 1
                    failed_dict[cnt_list[j]] += 1
                    if cnt_list[j]<4:
                        failed_dict_unanswerble[cnt_list[j]] += 1
        except:
            print(f'---------continue-------{i}-----------')
            continue
            

    return success_dict,failed_dict,answer_one,success_dict_unanswerble,failed_dict_unanswerble,s_cnt,f_cnt

if __name__ == '__main__':
    # txt_path = '../QA_agent_excord.txt'
    # txt_path = '../QA_agent_bert.txt'
    txt_path = '../eval_results/results_400_case/QA_agent_bert.txt'
    # txt_path = '../QA_agent_excord_ground_history_bugs.txt'
    # txt_path = '../eval_results/results_100_case/QA_agent_ham.txt'
    # txt_path = '../QA_agent_excord_ground_history.txt'
    # txt_path = '../QA_agent_GraphFlow.txt'
    s,f,n,s_n,f_n,s_c,f_c = parse_txt(txt_path=txt_path,answerble=False)
    pdb.set_trace()

"""
(4, 'Did Shula lose to any coaches?', ('Super Bowl losses', None, None)), (4, 'Who were some coaches that Don Shula had losing records against?', ('Tom Flores ( 1 - 6 ) Raymond Berry ( 3 - 8 ) ,', None, None)), (4, 'What were some other coaches that Don Shula had losing records against besides Tom Flores and Raymond Berry?', ('Walt Michaels ( 5 - 7 - 1 ) , and Vince Lombardi ( 5 - 8 ) .', None, None)), (4, 'Can you name all the coaches that Don Shula had losing records against?', ('Don Shula also had losing records against Tom Flores(1-6) Raymond Berry (3-8), Walt Michaels (5-7-1), and Vince Lombardi (5-8).', None, None))]###[True, True, True, True, False]
"""






        



