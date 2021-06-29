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
from sklearn.metrics import f1_score, r2_score
from sklearn.metrics import mean_squared_error
from skll.metrics import kappa as kpa
import evaluationutility as eu

#======================preprocessing========================
# file = open("1_bert_full_connected_responses.csv",encoding="utf-8")
# fileReader = csv.reader(file)
# print(len(list(fileReader)))
def encoding(scores):
    encodedList = []
    # print(scores[0:10])
    for single_score in range(len(scores)):
        list = [0,0,0,0,0]
        list[scores[single_score]] = 1
        # print(list)
        encodedList.append(list)
    # print(encodedList)
    return encodedList
#
#
# def rsquared(actual, predicted):
#     assert len(actual) == len(predicted)
#
#     ac = np.array(actual, dtype=np.float32).reshape((len(actual), -1))
#     pr = np.array(predicted, dtype=np.float32).reshape((len(predicted), -1))
#
#     na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()
#
#     return r2_score(ac[na], pr[na])
#
# def cohen_kappa_multiclass(actual, predicted):
#     assert len(actual) == len(predicted)
#
#     ac = np.array(actual).reshape((len(actual), -1))  #,dtype=np.float32
#     pr = np.array(predicted).reshape((len(predicted), -1))  #,dtype=np.float32
#
#     try:
#         na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()
#     except:
#         for i in ac:
#             print(i)
#
#         for i in ac:
#             print(np.any(np.isnan(i)))
#
#     if len(na) == 0:
#         return np.nan
#
#     aci = np.argmax(np.array(np.array(ac[na]) ), axis=1) #dtype=np.int32
#     pri = np.argmax(np.array(np.array(pr[na]) ), axis=1) #, dtype=np.float32
#
#     # for i in range(len(aci)):
#     #     print(aci[i],'--',pri[i],':',np.array(pr[na])[i])
#
#     return kpa(aci,pri)
#
#
#
# def auc(actual, predicted, average_over_labels=True, partition=1024.):
#     assert len(actual) == len(predicted)
#
#     ac = np.array(actual).reshape((len(actual),-1))   # Took out the dtype=np.float32 in the function
#     pr = np.array(predicted).reshape((len(predicted),-1))
#
#     na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()
#
#     ac = ac[na]
#     pr = pr[na]
#
#     label_auc = []
#     for i in range(ac.shape[-1]):
#         a = np.array(ac[:,i])
#         p = np.array(pr[:,i])
#         val = np.unique(a)
#         # if len(val) > 2:
#         #     print('AUC Warning - Number of distinct values in label set {} is greater than 2, '
#         #           'using median split of distinct values...'.format(i))
#         if len(val) == 1:
#             # print('AUC Warning - There is only 1 distinct value in label set {}, unable to calculate AUC'.format(i))
#             label_auc.append(np.nan)
#             continue
#
#         pos = np.argwhere(a[:] >= np.median(val))
#         neg = np.argwhere(a[:] < np.median(val))
#
#         # print(pos)
#         # print(neg)
#
#         p_div = int(np.ceil(len(pos)/partition))
#         n_div = int(np.ceil(len(neg)/partition))
#
#         # print(len(pos), p_div)
#         # print(len(neg), n_div)
#
#         div = 0
#         for j in range(int(p_div)):
#             p_range = list(range(int(j * partition), int(np.minimum(int((j + 1) * partition), len(pos)))))
#             for k in range(n_div):
#                 n_range = list(range(int(k * partition), int(np.minimum(int((k + 1) * partition), len(neg)))))
#
#
#                 eq = np.ones((np.alen(neg[n_range]), np.alen(pos[p_range]))) * p[pos[p_range]].T == np.ones(
#                     (np.alen(neg[n_range]), np.alen(pos[p_range]))) * p[neg[n_range]]
#
#                 geq = np.array(np.ones((np.alen(neg[n_range]), np.alen(pos[p_range]))) *
#                                p[pos[p_range]].T >= np.ones((np.alen(neg[n_range]),
#                                                              np.alen(pos[p_range]))) * p[neg[n_range]],
#                                dtype=np.float32)
#                 geq[eq[:, :] == True] = 0.5
#
#                 # print(geq)
#                 div += np.sum(geq)
#                 # print(np.sum(geq))
#                 # exit(1)
#
#         label_auc.append(div / (np.alen(pos)*np.alen(neg)))
#         # print(label_auc)
#
#     if average_over_labels:
#         return np.nanmean(label_auc)
#     else:
#         return label_auc
#
#
# def rmse(actual, predicted, average_over_labels=True):
#     assert len(actual) == len(predicted)
#
#     ac = np.array(actual, dtype=np.float32).reshape((len(actual), -1))
#     pr = np.array(predicted, dtype=np.float32).reshape((len(predicted), -1))
#
#     na = np.argwhere([not np.any(np.isnan(i)) for i in ac]).ravel()
#
#     if len(na) == 0:
#         return np.nan
#
#     ac = ac[na]
#     pr = pr[na]
#
#     label_rmse = []
#     for i in range(ac.shape[-1]):
#         dif = np.array(ac[:, i]) - np.array(pr[:, i])
#         sqdif = dif**2
#         mse = np.nanmean(sqdif)
#         label_rmse.append(np.sqrt(mse))
#
#
#     if average_over_labels:
#         return np.nanmean(label_rmse)
#     else:
#         return label_rmse

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
# print(sentence_embeddings.shape)


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
# data = np.array(data)
predictions = model.predict(data)
# predictions = [[0,0,0,0,1],[0,0,0,1,0],[0,0,0,1,0],[1,0,0,0,0],[0,0,0,1,0],[0,0,0,1,0]]

predictions = np.argmax(predictions,axis=1)
# print(predictions)
# predictions = list(predictions)
#predictions = model.predict(data).tolist()
#predictions = list(map(np.int32, predictions))
#print(predictions[0:2])
#np.argmax
#for row in predictions
encodingArr = encoding(training_score)
encodingPrediction = encoding(predictions)
print("rmse...")
print(eu.rmse(encodingArr,encodingPrediction))
print("auc...")


partition = [64., 128., 256., 512., 1024., 2048., 4096.]
for p in partition:
    try:
        a = eu.auc(encodingArr,encodingPrediction, p)
        print(a)
    except:
        print('{:^15}'.format('Failed'), end='')
print('')





# print(eu.auc(encodingArr,encodingPrediction))
print("kappa...")
print(eu.cohen_kappa_multiclass(encodingArr,encodingPrediction))
print("rsquared....")
print(eu.rsquared(encodingArr,encodingPrediction))

# model.save("LSTM_TEST")
