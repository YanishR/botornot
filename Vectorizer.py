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


    def getContentMatrix(self, cols, n):
        fm = np.zeros( (len(self.tweets)), cols)

        for i in range(len(self.tweets)):
            temp_col = 0
            ft = Featurizer(self.tweets[i])
            word_dict, ngrams = generateNgramID(n, 0)

            for word in ft.tweet: #NOTE DO THIS
                fm[i][temp_col] = 0
                temp_col += 1

            fm[i][temp_col] = ft.getAvgEmojis()
            fm[i][temp_col + 1] = ft.getNumURL()
            fm[i][temp_col + 2], fm[i][temp_col + 3] = ft.getNumTags()
            fm[i][temp_col + 4], fm[i][temp_col + 5], fm[i][temp_col + 6], dummy, dummy2 = ft.getPOSTaggedDistribution()
            fm[i][temp_col + 7] = ft.getNumTokens()


    def getStylisticMatrix(self, cols, n):
        fm = np.zeros( (len(self.tweets)), cols)

        for i in range(len(self.tweets)):
            temp_col = 0
            ft = Featurizer(self.tweets[i])
            word_dict, ngrams = generateNgramID(n, 1)

            for word in ft.tweet:
                fm[i][temp_col] =
                temp_col += 1

            fm[i][temp_col] = ft.getAvgNumPunct()
            fm[i][temp_col + 1] = ft.getAvgWordSize()
            fm[i][temp_col + 2] = ft.getVocabSize()
            dummy, dummy1, dummy2, fm[i][temp_col + 3], fm[i][temp_col + 4] = ft.getPOSTaggedDistribution()
            fm[i][temp_col + 5] = ft.getDigitFrequency()
            fm[i][temp_col + 6] = ft.getAvgHashTagLength()
            fm[i][temp_col +7] = ft.getLetterFrequency()
            fm[i][temp_col + 8] = ft.getAvgCapitalizations()
