import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import nltk
import tensorflow as tf
from tensorflow import keras
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os

#defs
MAX_NB_WORDS = 10000


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
    filtered_sentence = [w.lower() for w in word_tokens if not w.lower() in stop_words]
    cleaned_text = cleaned_text.join(filtered_sentence)
    updated_text.append(cleaned_text)


print(updated_text[0])
train_answer, test_answer, training_score, test_score = train_test_split(updated_text, score, test_size=0.2)

vocab_size = 0
#embedding dimension = ?
max_length = 0
padding_type = 'post' #post and same
trunc_type = 'post' #post
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_answer)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_answer)
data = pad_sequences(sequences, maxlen = max_length, padding = padding_type, truncating=trunc_type) #look at padding with a word over null value

testing_sequences= tokenizer.texts_to_sequences(test_answer)
testing_padded=pad_sequences(testing_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)

#glove?
embeddings = {}
f= open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    enbeddings_index[word] = coefs
f.close()
