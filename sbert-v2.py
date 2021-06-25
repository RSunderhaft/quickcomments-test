import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import CuDNNLSTM
import csv

#======================preprocessing========================
# file = open("1_bert_full_connected_responses.csv",encoding="utf-8")
# fileReader = csv.reader(file)
# print(len(list(fileReader)))
def encoding(scores):
    encodedList = []

    for single_score in range(len(scores)):
        list = [0,0,0,0,0]
        list[scores[single_score]] = 1
        encodedList.append(list)
    # print(encodedList)
    return encodedList



def rmse(actual, predicted, average_over_labels=True):
    assert len(actual) == len(predicted)

    ac = np.array(actual, dtype=np.float32).reshape((len(actual), -1))
    pr = np.array(predicted, dtype=np.float32).reshape((len(predicted), -1))

    na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()

    if len(na) == 0:
        return np.nan

    ac = ac[na]
    pr = pr[na]

    label_rmse = []
    for i in range(ac.shape[-1]):
        dif = np.array(ac[:, i]) - np.array(pr[:, i])
        sqdif = dif**2
        mse = np.nanmean(sqdif)
        label_rmse.append(np.sqrt(mse))


    if average_over_labels:
        return np.nanmean(label_rmse)
    else:
        return label_rmse

df= pd.read_csv("1_bert_full_connected_responses.csv")
#removed nan values
df.dropna(subset = ["cleaned_answer_text"],inplace = True)
score = df['score'].tolist()
cleaned_answer_text = df['cleaned_answer_text'].tolist()
stop_words = set(stopwords.words('english'))
updated_text = []
# remove stop words -- delete??
for val in cleaned_answer_text:
    cleaned_text = ' '
    word_tokens = word_tokenize(val)
    filtered_sentence = [w.lower() for w in word_tokens if not w.lower() in stop_words]
    cleaned_text = cleaned_text.join(filtered_sentence)
    updated_text.append(cleaned_text)

#========tokenize

train_answer, test_answer, training_score, test_score = train_test_split(updated_text, score, test_size=0.2)
vocab_size = 100#113289
#150448
embedding_dimension = 1024
max_length = 50
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_answer)

sequences = tokenizer.texts_to_sequences(train_answer)
data = pad_sequences(sequences, maxlen = max_length, padding = padding_type, truncating=trunc_type) #look at padding with a word over null value

testing_sequences= tokenizer.texts_to_sequences(test_answer)
testing_padded=pad_sequences(testing_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)

#embeddings...
# model = SentenceTransformer('paraphrase-distilroberta-base-v1')
# model = SentenceTransformer('bert-base-nli-mean-tokens')
model = SentenceTransformer('stsb-bert-large')
sentences = train_answer[0:100]
sentence_embeddings = model.encode(sentences)
print(sentence_embeddings.shape)


#model setup
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = tf.keras.Sequential([
tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dimension, input_length=max_length, weights=[sentence_embeddings],trainable=True),
tf.keras.layers.SpatialDropout1D(0.2),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
tf.keras.layers.Dropout(0.25),
tf.keras.layers.Dense(5,activation='softmax'),
])
#
#adamax, nadam, adam
opt = keras.optimizers.Adamax(learning_rate = 0.0001)
# model.compile(loss=MyLoss_Layer(),optimizer=opt, metrics=['accuracy'])
model.compile(loss='SparseCategoricalCrossentropy',optimizer=opt, metrics=['accuracy'])
#RMSE..
# model.compile(loss='mse',optimizer='sgd', metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()
model.fit(np.array(data), np.array(training_score),batch_size=128,epochs=1,validation_data=(np.array(testing_padded),np.array(test_score)), callbacks=[callback])
# print((model.predict(data) > 0.5).astype("int32"))
predictions = (model.predict(data) > 0.5).astype("int32")
encodingArr = encoding(training_score)
print(rmse(encodingArr,predictions))
# model.save("LSTM_TEST")
