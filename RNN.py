import torch.nn as nn
import torch
import gensim
import gensim.downloader as api

from tqdm import tqdm

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional
from tensorflow.keras import backend as K

vocab_size = 5000 # make the top list of words (common words)
embedding_dim = 32
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>' # OOV = Out of Vocabulary

# numpy.random.seed(7)

# glove_model = api.load("glove-twitter-25")  # load glove vectors
from Featurizer import *
from dataparser import *
from tweet import *



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
    self.model = None
    self.vocab_size = 5000

  def initializeTokenizer(self):
    tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(self.data)
    self.data_tokenizer = tokenizer

  def convert2vec(self, data):
    sequences = self.data_tokenizer.texts_to_sequences(data)
    padded_seq = pad_sequences(sequences, maxlen=max_length, 
      padding=padding_type, truncating=trunc_type)
    return padded_seq
  
  def embeddingMetrics(self):
    glove_model = api.load("glove-twitter-200")  # load glove vectors
    word_vectos = glove_model.wv
    self.vocab_size = len(self.data_tokenizer.word_index.items()) + 1
    embedding_matrix = np.zeros((self.vocab_size, max_length))
    for word,i in tqdm(self.data_tokenizer.word_index.items()):
      try:
        embedding_value = word_vectos[word]
      except:
        embedding_value = None
      if embedding_value is not None:
        embedding_matrix[i] = embedding_value
    return embedding_matrix

  def trainLSTM(self):
    def recall_m(y_true, y_pred):
      true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
      possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
      recall = true_positives / (possible_positives + K.epsilon())
      return recall

    def precision_m(y_true, y_pred):
      true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
      predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
      precision = true_positives / (predicted_positives + K.epsilon())
      return precision

    def f1_m(y_true, y_pred):
      precision = precision_m(y_true, y_pred)
      recall = recall_m(y_true, y_pred)
      return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    self.initializeTokenizer()
    train_padded = self.convert2vec(self.data)
    train_label = np.array(self.label)
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    # embedding_matrix = self.embeddingMetrics()
    # model.add(Embedding(self.vocab_size, max_length, weights = [embedding_matrix], input_length= max_length, trainable = False))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','AUC', f1_m, precision_m, recall_m])
    print(model.summary())
    model.fit(train_padded, train_label, validation_split=0.2, epochs=3, batch_size=64)
    self.model = model

  def evaluate(self, test_data, test_label):
    test_padded = self.convert2vec(test_data)
    test_label_new = np.array(test_label)
    scores = self.model.evaluate(test_padded, test_label_new, verbose=1)
    print(scores)
    print("Accuracy: %.2f%%" % (scores[1]*100))



if __name__ == "__main__":
    print("hello world!")
    electionTweets = "./data/2016_US_election_tweets_100k.csv"
    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    tweets = Data(electionTweets, electionTrolls)
    x_train, x_test, y_train, y_test = tweets.getSplitDataDL(.3)
    # print(x_train[:10])
    # print(y_train[:10])
    train_data = []
    test_data = []
    for t in x_train:
      T = Tweet(t)
      train_data.append(T.preprocess())
    for t in x_test:
      T = Tweet(t)
      test_data.append(T.preprocess())
    # print(train_data[:10])
    print(len(train_data), len(y_train))
    print(len(test_data), len(y_test))
    myRNN = RNN(train_data, y_train)
    myRNN.trainLSTM()
    myRNN.evaluate(test_data, y_test)







