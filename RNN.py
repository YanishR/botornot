import gensim.downloader as api

from tqdm import tqdm

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation, Embedding, Bidirectional
from tensorflow.keras import backend as K

from dataparser import *
from tweet import *

# hyperparameters
embedding_dim = 32 # embedding dimension
max_length = 200 # tweet max length 
trunc_type = 'post' # truncate at the end
padding_type = 'post' # padding at the end
oov_tok = '<OOV>' # OOV = Out of Vocabulary

class RNN():
  '''Construct RNN model
  Input: data  - a list of string (tweets)
         label - a list of binary number indicating categories
  '''
  def __init__(self, electionTweets, electionTrolls, testSpilt=0.3):
    self.dataset = Data(electionTweets, electionTrolls)
    self.testSplit = testSpilt
    self.data_tokenizer = None
    self.model = None
    self.vocab_size = 5000
    self.initialize()

  def initialize(self):
    x_train, x_test, y_train, y_test = self.dataset.getSplitDataDL(self.testSplit)
    train_data = [] # for preprocess
    test_data = []
    for t in x_train:
      T = Tweet(t)
      train_data.append(T.preprocess())
    for t in x_test:
      T = Tweet(t)
      test_data.append(T.preprocess())
    # print(len(train_data), len(y_train))
    # print(len(test_data), len(y_test))
    self.trainData = train_data
    self.trainLabel = y_train
    self.testData = test_data
    self.testLabel = y_test
    

  '''Initialized the training tokenizer'''
  def initializeTokenizer(self):
    tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(self.trainData)
    self.data_tokenizer = tokenizer

  '''Convert string data to number vectors with padding and truncation
  Input: data - a list of string (tweets)
  Return: vectors for data - a list of vectors
  '''
  def convert2vec(self, data):
    sequences = self.data_tokenizer.texts_to_sequences(data)
    padded_seq = pad_sequences(sequences, maxlen=max_length, 
      padding=padding_type, truncating=trunc_type)
    return padded_seq
  
  ''' Get embedding Metrics from GloVe embeddings
  Used only for GloVe embeddings
  Return: embedding matrix
  '''
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

  '''Train LSTM model'''
  def trainLSTM(self):
    '''Recall, Precision, F1-Score computation. 
    Keras removed them from core metrics'''
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
    train_padded = self.convert2vec(self.trainData)
    train_label = np.array(self.trainLabel)
    model = Sequential()
    model.add(Embedding(self.vocab_size, embedding_dim))
    # embedding_matrix = self.embeddingMetrics() # only used for GloVe embeddings
    # model.add(Embedding(self.vocab_size, max_length, weights = [embedding_matrix], 
    # input_length= max_length, trainable = False)) # only used for GloVe embeddings
    # model.add(Dropout(0.2)) # prevent overfitting
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','AUC', f1_m, precision_m, recall_m])
    print(model.summary())
    model.fit(train_padded, train_label, epochs=1, validation_split=0.2, batch_size=16, verbose=0)
    self.model = model

  '''Predict and Evaluate using trained LSTM model, Print scores.
  Input: test_data - test set data, a list of string (tweets)
         test_label - test set label, a list of binary number indicating categories
  Return: scores - a list of metrics results
  '''
  def evaluate(self):
    test_padded = self.convert2vec(self.testData)
    test_label_new = np.array(self.testLabel)
    scores = self.model.evaluate(test_padded, test_label_new, verbose=0)
    # print(scores)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return scores

  def run(self):
    print("Running LSTM experiments")
    self.trainLSTM()
    scores = self.evaluate()
    print(scores)

    
    
