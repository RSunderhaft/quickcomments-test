import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


df= pd.read_csv("/Users/robert/Desktop/1_bert_full_connected_responses.csv")
#removed nan values
df.dropna(subset = ["cleaned_answer_text"],inplace = True)
sentences = df['cleaned_answer_text'].tolist()
scores = df['score'].tolist()

model_name = 'bert-base-nli-mean-tokens' #possibly test out other model names

model = SentenceTransformer(model_name)
sentence_vecs = model.encode(sentences[0:100])


sims_of_sentence = cosine_similarity([sentence_vecs[99]], sentence_vecs[:99])

sims_of_sentence = (sims_of_sentence[0]).tolist()
max_sim = max(sims_of_sentence)
index_of_max = sims_of_sentence.index(max_sim)

print("score input sentence: " + str(scores[99]) + "\nsentence: " + str(sentences[99]))
print("score output sentence: " + str(scores[index_of_max]) + "\nsentence: " + str(sentences[index_of_max]))
