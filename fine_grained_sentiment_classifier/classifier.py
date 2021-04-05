import json

from torch import nn
from transformers import BertModel

with open("./config_fine_grained.json") as json_file:
    config = json.load(json_file)


class FineGrainedSentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(FineGrainedSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config["BERT_MODEL"])
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)