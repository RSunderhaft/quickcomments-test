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
from tensorflow.keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
from sentence_transformers import SentenceTransformer

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

train_answer, test_answer, training_score, test_score = train_test_split(updated_text, score, test_size=0.2)

vocab_size = 8000
embedding_dimension = 100
max_length = 100  
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<OOV>"
MAX_NB_WORDS = 10000


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_answer)
# word_index = tokenizer.word_index
word_index = MAX_NB_WORDS
sequences = tokenizer.texts_to_sequences(train_answer)
data = pad_sequences(sequences, maxlen = max_length, padding = padding_type, truncating=trunc_type) #look at padding with a word over null value

testing_sequences= tokenizer.texts_to_sequences(test_answer)
testing_padded=pad_sequences(testing_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)

# embeddingsDict = {}
# f= open('glove.6B.100d.txt')
# for line in f:
#     value = line.split()
#     word = value[0]
#     vector= np.asarray(value[1:],"float32")
#     embeddingsDict[word] = vector
# f.close()
#
# embeddingsMatrix = np.array([embeddingsDict[i] for i in embeddingsDict.keys()] )
# #
# hits = 0
# misses = 0
# num_tokens = vocab_size + 2
#
# embedding_matrix = np.zeros((num_tokens,embedding_dimension))
# for word,i in word_index.items():
#     embedding_vector = embeddings.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
#         hits+=1
#     else:
#         misses+=1

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = tf.keras.Sequential([
tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dimension,input_length=max_length, mask_zero=True, embeddings_initializer=embeddings),
# tf.keras.layers.LSTM()
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(5,activation='softmax')
#tf.keras.layers.Dense(1,activation='relu') #todo activation
])
# loss_fn = keras.losses.CategoricalCrossentropy()  cohen_kappa
#SparseCategoricalCrossentropy

#adamax, nadam, adam
model.compile(loss='SparseCategoricalCrossentropy',optimizer='adamax', metrics=['accuracy']) #metrics = auc ; kappa
model.summary()
model.fit(np.array(data), np.array(training_score),batch_size=128,epochs=50,validation_data=(np.array(testing_padded),np.array(test_score)), callbacks=[callback])
