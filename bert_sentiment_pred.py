#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:40:37 2020

@author: yuyu
"""
#source $HOME/.cargo/env
#!pip install -qq transformers

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
#from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 512
class_names =['negative','positive']

class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.5)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

model = SentimentClassifier(2)
model.load_state_dict(torch.load('best_model_state_512_lr5e-6.bin',map_location=torch.device('cpu')))
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME,do_lower_case=True)
#input text
review_text = "I hate completing reading this book! Worst Book Ever!!!"

def prediction(text,model,class_name):
    encoding = tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=MAX_LEN,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
      truncation=True
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    return class_name[prediction]

output_class = prediction(review_text,model,class_names)




#print(f'Review text: {review_text}')
#print(f'Sentiment  : {class_names[prediction]}')

