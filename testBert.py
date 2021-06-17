import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant

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
print(type(train_answer))

vocab_size = 8000
embedding_dimension = 100
max_length = 100
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<OOV>"
MAX_NB_WORDS = 10000
#embeddings...
# model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
# model = SentenceTransformer('paraphrase-distilroberta-base-v1')
model = SentenceTransformer('stsb-bert-large')
#Our sentences we like to encode
# sentences = ['This framework generates embeddings for each input sentence','Sentences are passed as a list of string.','The quick brown fox jumps over the lazy dog.'] #answer text
# sentences = updated_text[0:100]
sentences = train_answer[0:200]
#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = tf.keras.Sequential([
# tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dimension,input_length=max_length, mask_zero=True, embeddings_initializer= keras.initializers.Constant(sentence_embeddings)),
tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dimension,embeddings_initializer= Constant(sentence_embeddings), trainable=False),
tf.keras.layers.GlobalAveragePooling1D(),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(5,activation='softmax')
])

#adamax, nadam, adam
model.compile(loss='SparseCategoricalCrossentropy',optimizer='adamax', metrics=['accuracy'])
model.summary()
model.fit(np.array(data), np.array(training_score),batch_size=128,epochs=50,validation_data=(np.array(testing_padded),np.array(test_score)), callbacks=[callback])
