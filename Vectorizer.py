import Featurizer
import numpy as np
from nltk import ngrams

class Vectorizer():

    def __init__(self, tweets):
        self.tweets = tweets
        pass

    def getNgram(self, n):
        g = {}
        for tweet in self.tweets:

            for seq in ngrams(tweet.split(), n):
                s = ""
            for w in seq:
                s += w + " "

            s = s[:-1]
            g[s] += 1 if s in g else 1
        return g

    def getCharNgram(self, n):
        g = {}
        for tweet in self.tweets:

            for seq in ngrams(tweet, n):
                s = ""

                for c in seq:
                    s += c

                g[s] += 1 if s in g else 1
                return g

    def generateNgramID(self, n, type):
        ind = 0
        ngrams = getNgram(n) if type == 0 else getCharNgram(n)
        word_dict = {}

        for word in ngrams:
            if word not in word_dict:
                word_dict[word] = ind
                ind += 1

        return word_dict, ngrams

    def getContentMatrix(self, cols, n, type):
        fm = np.zeros( (len(self.tweets)), cols)
        for i in range(len(self.tweets)):
            temp_col = 0
            ft = Featurizer(self.tweets[i])
            word_dict, ngrams = generateNgramID(n, type)
            for word in ft.:
                fm[i][temp_col] =






    def getStylisticMatrix(self):
        pass
