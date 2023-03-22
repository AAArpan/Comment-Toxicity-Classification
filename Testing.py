import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv(r"D:\ML\train.csv")
X = df['comment_text']
Y = df[df.columns[2:]].values

sentences = []
for s in X:
  sentences.append(s)

vocab_size = 200000
max_length = 1800
trunc_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

model_lstm = tf.keras.models.load_model("D:\ML\model_lstm.h5")

classes = ['toxic',  'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] 

test_review = ["Freaking hate you! Bastard"]

sequences = tokenizer.texts_to_sequences(test_review)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

prob = model_lstm.predict(padded)
print("Comment is ", round(prob[0][0]*100,2), "% toxic")
print("Comment is ", round(prob[0][1]*100,2), "% severe_toxic")
print("Comment is ", round(prob[0][2]*100,2), "% obscene")
print("Comment is ", round(prob[0][3]*100,2), "% threat")
print("Comment is ", round(prob[0][4]*100,2), "% insult")
print("Comment is ", round(prob[0][5]*100,2), "% identity_hate")
