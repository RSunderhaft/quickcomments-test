import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math

df= pd.read_csv("1_bert_full_connected_responses.csv")

#removed nan values
df.dropna(subset = ["cleaned_answer_text"],inplace = True)

score = df['score'].tolist()
cleaned_answer_text = df['cleaned_answer_text'].tolist()
stop_words = set(stopwords.words('english'))
updated_text = []

# remove stop words
for val in cleaned_answer_text:
    cleaned_text = ''
    word_tokens = word_tokenize(val)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    cleaned_text = cleaned_text.join(filtered_sentence)
    updated_text.append(cleaned_text)

train_answer, test_answer, training_score, test_score = train_test_split(updated_text, score, test_size=0.2)
