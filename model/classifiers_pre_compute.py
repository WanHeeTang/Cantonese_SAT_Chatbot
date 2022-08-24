import pytorch_lightning as pl
import textdistance as td
import numpy as np
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import re
import nltk
import pandas as pd
import math
nltk.download("stopwords")
from nltk.corpus import stopwords

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoModelWithLMHead,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)


emo_model = BertForSequenceClassification.from_pretrained('/home/wanhee/Cantonese_SAT_Chatbot/model/Emotion Classification/BestBERTEmotion')
emo_tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

#load emotion classifier (BestBERTEmotion.pt)
with torch.no_grad():
    emo_model.to(torch.device('cpu'))
    emo_model.load_state_dict(torch.load('/home/wanhee/Cantonese_SAT_Chatbot/model/Emotion Classification/BestBERTEmotion/BestBERTEmotion.pt', map_location=torch.device('cpu')), strict=False)


# simple tokenizer + stemmer
regextokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stemmer = nltk.stem.PorterStemmer()
# readcsv
df = pd.read_csv("/home/wanhee/Cantonese_SAT_Chatbot/Dataset/Empathy Classification/Additional Dataset/merge_neutral.csv", encoding='UTF-8')



def get_emotion(text):
    X_test_tokenized = emo_tokenizer(text, padding=True, truncation=True, max_length=512)

    b_input_ids= torch.tensor(X_test_tokenized['input_ids']).unsqueeze(0)
    b_attention_mask = torch.tensor(X_test_tokenized['attention_mask']).unsqueeze(0)

    with torch.no_grad():
        outputs = emo_model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)

    logits = outputs[0]
    print(logits)
    logits = logits.detach().numpy()
    predict_label = np.argmax(logits, axis=1).flatten()
    print(predict_label)

    label = " "

    for i in predict_label:
        if i == 0:
            label = "唔開心"
        elif i == 1:
            label = "嬲"
        elif i == 2:
            label = "擔心"
        else:
            label = "開心"
        print(label)
    
    return label


def get_distance(s1, s2):
    '''
    Computes a distance score between utterances calculated as the overlap
    distance between unigrams, plus the overlap distance squared over bigrams,
    plus the overlap distance cubed over trigrams, etc (up to a number of ngrams
    equal to the length of the shortest utterance)
    '''
    s1 = re.sub(r'[^\w\s]', '', s1.lower())  # preprocess
    s2 = re.sub(r'[^\w\s]', '', s2.lower())
    s1_ws = regextokenizer.tokenize(s1)  # tokenize to count tokens later
    s2_ws = regextokenizer.tokenize(s2)
    # the max number of bigrams is the number of tokens in the shorter sentence
    max_n = len(s1_ws) if len(s1_ws) < len(s2_ws) else len(s2_ws)
    ngram_scores = []
    for i in range(1, max_n+1):
        s1grams = nltk.ngrams(s1.split(), i)
        s2grams = nltk.ngrams(s2.split(), i)
        # we normalize the distance score to be a value between 0 and 10, before raising to i
        ngram_scores.append(
            (td.overlap.normalized_distance(s1grams, s2grams))**i)
    normalised_dis = sum(ngram_scores)/(max_n)  # normalised
    return normalised_dis


def compute_distances(sentence, dataframe):
    '''
    Computes a list of distances score between an utterance and all the utterances in a dataframe
    '''
    distances = []
    for index, row in dataframe.iterrows():
        # assuming the dataframe column is called 'sentences'
        df_s = dataframe['sentences'][index]
        distance = get_distance(df_s.lower(), sentence)
        distances.append(distance)
    return distances


def novelty_score(sentence, dataframe):
    '''
    Computes the mean of the distances beween an utterance
    and each of the utterances in a given dataframe
    '''
    if dataframe.empty:
        score = 1.0
    else:
        d_list = compute_distances(sentence, dataframe)
        d_score = sum(d_list)
        score = d_score / len(d_list)
    return round(score, 2)

    
    
# def get_sentence_score(sentence, dataframe):
#     '''
#     Calculates how fit a sentence is based on its weighted empathy, fluency
#     and novelty values
#     '''
#     tmp_df = df[df["response"]==sentence]
#     tmp = tmp_df.iloc[0]
#     empathy = float(tmp.empathy_score)
#     fluency = float(tmp.fluency_score)
#     novelty = novelty_score(sentence, dataframe)
#     sentiment = float(tmp.sentiment_score)
#     score = 1.5*empathy + fluency + 2*novelty +0.3*sentiment
#     return score

def get_sentence_score(sentence, dataframe):
  '''
  Calculates how fit a sentence is based on its weighted empathy, fluency
  and novelty values
  '''
  tmp_df = df[df["response"]==sentence]
  tmp = tmp_df.iloc[0]
  print(tmp)
  empathy = float(tmp.empathy_score)
  fluency = (math.log(float(tmp.fluency_score)) + 5) / 5
  novelty = novelty_score(sentence, dataframe)
  sentiment = (math.log(1 - float(tmp.sentiment_score) + 0.00001) + 6) / 6
  score = 2*empathy + fluency + 1.5*novelty + sentiment

  return score
