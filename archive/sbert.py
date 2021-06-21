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
    cleaned_text = ' '
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
#sentence embeddings
model = SentenceTransformer('paraphrase-distilroberta-base-v')
#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence','Sentences are passed as a list of string.','The quick brown fox jumps over the lazy dog.'] #answer text
print("one")
#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)
print("two")
#Print the embeddings
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = tf.keras.Sequential([
tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dimension,input_length=max_length, mask_zero=True, embeddings_initializer=embeddings),
tf.keras.layers.GlobalAveragePooling1D(),
tf.keras.layers.Dense(5,activation='softmax')
])

#adamax, nadam, adam
model.compile(loss='SparseCategoricalCrossentropy',optimizer='adamax', metrics=['accuracy'])
model.summary()
model.fit(np.array(data), np.array(training_score),batch_size=128,epochs=50,validation_data=(np.array(testing_padded),np.array(test_score)), callbacks=[callback])
