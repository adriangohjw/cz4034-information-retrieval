import json

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from sentiment_analysers.classifier import EmotionClassifier, FineGrainedSentimentClassifier, SentimentClassifier

with open("./config_emotion.json") as json_file:
    emotion_config = json.load(json_file)

with open("./config_fine_grained.json") as json_file:
    fg_config = json.load(json_file)

with open("./config.json") as json_file:
    config = json.load(json_file)

class EmotionModel:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(emotion_config["BERT_MODEL"])

        classifier = EmotionClassifier(len(emotion_config["SENTIMENT"]))
        classifier.load_state_dict(
            torch.load(emotion_config["PRE_TRAINED_MODEL"], map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=emotion_config["MAX_SEQUENCE_LEN"],
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(self.classifier(input_ids, attention_mask), dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()
        return (
            emotion_config["SENTIMENT"][predicted_class],
            confidence,
            dict(zip(emotion_config["SENTIMENT"], probabilities)),
        )

class FineGrainedModel:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(fg_config["BERT_MODEL"])

        classifier = FineGrainedSentimentClassifier(len(fg_config["SENTIMENT"]))
        classifier.load_state_dict(
            torch.load(fg_config["PRE_TRAINED_MODEL"], map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=fg_config["MAX_SEQUENCE_LEN"],
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(self.classifier(input_ids, attention_mask), dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()
        return (
            fg_config["SENTIMENT"][predicted_class],
            confidence,
            dict(zip(fg_config["SENTIMENT"], probabilities)),
        )

class Model:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(config["BERT_MODEL"])

        classifier = SentimentClassifier(len(config["SENTIMENT"]))
        classifier.load_state_dict(
            torch.load(config["PRE_TRAINED_MODEL"], map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=config["MAX_SEQUENCE_LEN"],
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(self.classifier(input_ids, attention_mask), dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()
        return (
            config["SENTIMENT"][predicted_class],
            confidence,
            dict(zip(config["SENTIMENT"], probabilities)),
        )

emotion_model = EmotionModel()
fg_model = FineGrainedModel()
model = Model()

def get_emotion_model():
    return emotion_model

def get_fine_grained_model():
    return fg_model

def get_model():
    return model