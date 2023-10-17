import os
# import pandas as pd
# import numpy as np
from collections import defaultdict
import pdb
## 平均成功轮数
## 不成功比例
## 轮数占比
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
    s_cnt = 0
    f_cnt = 0
    for i in data:
        try:
            # pdb.set_trace()
            history_str,states_str = i.split('###')
            history_ = eval(history_str)
            history = [(i[0]-1,i[1],i[2]) for i in history_]
            states = eval(states_str)
            cnt_list = [0]*len(states)
            turn_answers = [[(0,'')]]
            flag = False
            for j in history:
                if 'Your answer is right' in j[1]:
                    # print('hi')
                    answer_one += 1
                    continue
                try:
                    cnt_list[int(j[0])]+=1
                except Exception as e:
                    print(e)
                    # pdb.set_trace()
                finally:
                    # pdb.set_trace()
                    if j[2][0]=='unknown':
                        flag = True
                    if int(j[0]) == turn_answers[-1][0][0]:
                        turn_answers[-1].append((int(j[0]),j[2][0]))
                    else:
                        turn_answers.append([])
                        turn_answers[-1].append((int(j[0]),j[2][0]))
            for j in range(len(states)):
                # print(len(turn_answers),j)
                if states[j]:

                    success_dict[cnt_list[j]] += 1
                    if 'unknown' in turn_answers[j][-1][1]:
                        # print('hi')
                        success_dict_unanswerble[cnt_list[j]] += 1
                    if 'unknown' in turn_answers[j][0][1]:
                        s_cnt += 1
                else:
                    if 'unknown' in turn_answers[j][0][1]:
                        f_cnt += 1
                    failed_dict[cnt_list[j]] += 1
                    if cnt_list[j]<4:
                        failed_dict_unanswerble[cnt_list[j]] += 1
                if flag:
                    # pdb.set_trace()
                    flag = False
        except:
            print(f'---------continue-------{i}-----------')
            continue
            

    return success_dict,failed_dict,answer_one,success_dict_unanswerble,failed_dict_unanswerble,s_cnt,f_cnt

if __name__ == '__main__':
    # txt_path = '../QA_agent_excord.txt'
    # txt_path = '../QA_agent_bert.txt'
    # txt_path = '../QA_agent_excord_ground_history_bugs.txt'
    txt_path = '../eval_results/case_200/QA_agent_bert_ground_truth.txt'
    # txt_path = '../QA_agent_excord_ground_history.txt'
    # txt_path = '../QA_agent_GraphFlow.txt'
    s,f,n,s_n,f_n,s_c,f_c = parse_txt(txt_path=txt_path,answerble=False)
    pdb.set_trace()

"""
bert


(Pdb) p s

generate_setting
defaultdict(<class 'int'>, {1: 435, 2: 129, 3: 27, 4: 9})
对话次数810  对话条数600 平均对话轮数1.35
(Pdb) p f
defaultdict(<class 'int'>, {2: 75, 4: 46, 3: 27})
对话条数 148
(Pdb) p n
160


ground_setting

(Pdb) p s
defaultdict(<class 'int'>, {1: 448, 3: 25, 2: 125, 4: 9})
对话次数809  对话条数607 平均对话轮数1.33
(Pdb) p f
defaultdict(<class 'int'>, {4: 40, 2: 72, 3: 29})
141
(Pdb) p n
151

excord

(Pdb) p s
defaultdict(<class 'int'>, {1: 508, 2: 89, 3: 24, 4: 8})
对话次数796  对话条数641 平均对话轮数1.24
(Pdb) p f
defaultdict(<class 'int'>, {4: 36, 3: 26, 2: 57})
107
(Pdb) p n
162



graphflow

(Pdb) p s
defaultdict(<class 'int'>, {1: 411, 2: 107, 4: 21, 3: 36})
对话次数817 对话条数575 平均对话轮数1.51
(Pdb) p f
defaultdict(<class 'int'>, {4: 98, 2: 52, 3: 23})
173
(Pdb) p n
160

[1379,384,140,68]
[1186,563,140,49]
[1492,347,95,38]
[195,92,45]
[233,86,22]
"""






        



