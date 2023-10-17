from transformers import BertModel,BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch
import pdb

class Detector(object):
    def __init__(self,model) -> None:
        self.model = model
    def response():
        pass

def detect():
    """
    detect invalid question could 
    """
    pass

def rewrite(pred_answer,gold_answer):
    """
    if question is invalid, then rewrite the question, give some response about whether answer is right.
    """
    if pred_answer == gold_answer:
        return ''
    elif True:
        pass