import torch.nn as nn
import torch
import gensim
import gensim.downloader as api
glove_model = api.load("glove-twitter-25")  # load glove vectors
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
3. send to Glove word embeddings; return vectors
'''

'''
use vectors to form tensors


'''
# rnn = nn.LSTM(10, 10, 2) # input size, hidden size, num_layers

# input = torch.randn(5, 3, 10)
# print(input)
# h0 = torch.randn(2, 3, 10)
# c0 = torch.randn(2, 3, 10)
# output, (hn, cn) = rnn(input, (h0, c0))

# print(len(output))


# m = nn.ReLU()
# input = torch.randn(2)
# print(input)
# output = m(input)
# print(output)

def word_embeddings(features):
  sentences = []
  for f in features:
    sentences.append(f.tokens)
  # model = gensim.models.Word2Vec(glove_model)
  # model.build_vocab(sentences, update=True)
  # model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
  # print(model.wv)
  # print(glove_model)
  # return model.mv

tweet = " aaa bb cccc dd e y zz"
F = []
F.append(Featurizer(tweet))
word_embeddings(F)

# def LSTM(features):
#   # form tensor 
#   #...
#   # 
#   rnn = nn.LSTM(input)




