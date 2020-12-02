import torch.nn as nn
import torch
import gensim
import gensim.downloader as api

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional

vocab_size = 5000 # make the top list of words (common words)
embedding_dim = 32
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>' # OOV = Out of Vocabulary
training_portion = .8

# numpy.random.seed(7)

# glove_model = api.load("glove-twitter-25")  # load glove vectors
from Featurizer import *


'''
1. preprocessing words:
  - We replace occurrences of hashtags, URLs, numbers and user mentions with 
  the tags “<hashtag>", “<url>", “<number>", or ‘<user>".
  • Similarly, most common emojis are replaced with “<smile>", “<heart>", 
  “<lolface>", “<neutralface>" or “<angryface>",depending on the specific emoji.
  • For words written in upper case letters or for words containing more than 2 
  repeated letters, a tag denoting that is placed after the occurrence of the word. 
  For example, the word “HAPPY" would by replaced by two tokens, “happy" and “<allcaps>".
  • All tokens are converted to lower case.
2. tokenize
'''

'''
1. word embeddings; return vectors
use vectors to form tensors

'''

class RNN():
  def __init__(self, data, label):
    self.data = data
    self.label = label
    self.data_tokenizer = None
    self.label_tokenizer = None
    self.model = None

  def initializeTokenizer(self):
    tokenizer = Tokenizer(num_words=vocab_size, ovv_token=oov_tok)
    tokenizer.fit_on_texts(self.data)
    self.data_tokenizer = tokenizer
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(self.label)
    self.label_tokenizer = label_tokenizer

  def convert2vec(self):
    sequences = self.data_tokenizer.texts_to_sequences(self.data)
    padded_seq = pad_sequences(sequences, maxlen=max_length, 
      padding=padding_type, truncating=trunc_type)
    label_seq = np.array(self.label_tokenizer.texts_to_sequences(self.label))
    return padded_seq, label_seq

  def trainLSTM(self):
    train_padded, training_label_seq = self.convert2vec()
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
    print(model.summary())
    model.fit(train_padded, training_label_seq, epochs=3, batch_size=64)
    self.model = model


if __name__ == "__main__":
    print("hello world!")




