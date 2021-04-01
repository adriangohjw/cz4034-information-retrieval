#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import torch
from tqdm import tqdm
torch.cuda.empty_cache()
import numpy as np
from sklearn.metrics import f1_score

from torch.utils.data import TensorDataset
from transformers import BertTokenizer

from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time


labels_dict = {'ANGRY':0, 
          'DISGUST':1, 
          'FEAR':2, 
          'HAPPY':3, 
          'SAD':4, 
          'SURPRISE':5}

# Enter your sequence here:
sequence = [
    "I'M SO GONNA KILL YOU YOU PRICK!!!",
    "Yucks.. gross...",
    "I'm hiding at home till this is over.",
    'I LOVE DONUTS',
    "A pity... a pity.",
    "WHAT?! When did this happen!!!"    
]

# Create tokenizer object
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)

# Create model and load weights
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels = 6,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
PATH = './bert emotion/BERT_emotion_epoch_10.pt'
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')), strict=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def evaluate(valid_dataloader):
    model.eval()

    total_eval_loss = 0
    y_hat, y = [], []

    for batch in tqdm(valid_dataloader):
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids' : batch[0],
                'attention_mask': batch[1],
                'labels' : batch[2]
                }
        
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        y_hat.append(logits)
        y.append(label_ids)

    avg_eval_loss = total_eval_loss/len(valid_dataloader) 

    y_hat = np.concatenate(y_hat, axis=0)
    y = np.concatenate(y, axis=0)
            
    return avg_eval_loss, y_hat, y

valid_encode = tokenizer.batch_encode_plus(
    sequence,
    padding=True,
    truncation=True,
    add_special_tokens=True,
    max_length=256,   
    return_tensors='pt'
)
valid_input = valid_encode['input_ids']
valid_attention = valid_encode['attention_mask']
# The line below is a hack. I'm not entirely sure how to work
# with PyTorch to predict using purely a sequence.
labels = [0]*len(valid_input)
valid_labels = torch.tensor(labels)

valid_data = TensorDataset(valid_input,
                          valid_attention,
                        valid_labels)

valid_dataloader = DataLoader(valid_data,
                              sampler = SequentialSampler(valid_data),
                              batch_size = 8)

# Actual prediction happens here
_, predictions, actual = evaluate(valid_dataloader)

# Printing of labels sequence 
labs = {v:k for k,v in labels_dict.items()}
for i in tqdm(range(len(predictions))):
    print('Label: {}, Sequence:{}'.format(labs[np.argmax(predictions[i])], sequence[i]))
    print()



