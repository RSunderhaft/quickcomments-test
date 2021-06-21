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
vocab_size = 113289
embedding_dimension = 1024
max_length = 100
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
model = SentenceTransformer('stsb-bert-large')
sentences = train_answer
sentence_embeddings = model.encode(sentences)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = tf.keras.Sequential([
# tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dimension,input_length=max_length, mask_zero=True, embeddings_initializer= keras.initializers.Constant(sentence_embeddings)),
tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dimension, input_length=max_length, weights=[sentence_embeddings],trainable=False),
tf.keras.layers.GlobalAveragePooling1D(),
tf.keras.layers.Flatten(),
#kernel_regularizer='l2'
tf.keras.layers.Dense(5,activation='softmax'),
])

#adamax, nadam, adam
opt = keras.optimizers.Adamax(learning_rate = 0.01)
model.compile(loss='SparseCategoricalCrossentropy',optimizer=opt, metrics=['accuracy'])
model.summary()

model.fit(np.array(data), np.array(training_score),batch_size=128,epochs=50,validation_data=(np.array(testing_padded),np.array(test_score)), callbacks=[callback])
