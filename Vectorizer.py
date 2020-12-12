from tweet import Tweet
from dataparser import Data
import numpy as np

"""
Vectorizer:
    Class that transforms troll Tweets and real Tweets into
    a content and stylistic matrix
"""
class Vectorizer():

    def __init__(self, data):
        self.data = data

    """
    genGram(): Generate dictionary of ngrams or charngrams depending on
    char's boolean value
    Input:  n    : Integer describing n for word/character n grams
            char : boolean value determining whether this is a word ngram
            or char n gram
    Output: dict g: dictionary of ngrams or char ngrams and their occurences
    """
    def genGram(self, n, char=False):
        g = {} # Declare dictionary

        for tweet in self.data.getAllTweets():
            gram = tweet.getNgram(n) if char==False else tweet.getCharNgram(n)
            for ngram in gram:
                if ngram in g:
                    g[ngram] += gram[ngram]
                else:
                    g[ngram] = gram[ngram]
        final_g = {}

        for gram in g:
            if g[gram] >= 5:
                final_g[gram] = g[gram]
        return final_g

    """
    genNgram(): Returns dictionary of ids for each ngram and a dictionary of ngrams
    Input:  n    : Integer, Describes n for word/character n grams
            type : Integer, 0 for word Ngram, anything other integer for character Ngrams
    Output: word_dict: Returns dictionary of ids for each ngram
            ngrams: dictionary of word/character ngrams depending on the type
    """
    def genNgram(self, n):
        return self.genGram(n)

    """
    generateNCharNgram(): Returns dictionary of ids for each ngram and a dictionary of ngrams
    Input:  n    : Integer, Describes n for word/character n grams
            type : Integer, 0 for word Ngram, anything other integer for character Ngrams
    Output: word_dict: Returns dictionary of ids for each ngram
            ngrams: dictionary of word/character ngrams depending on the type
    """
    def genCharNgram(self, n):
        return self.genGram(n, True)

    """
    generateNgramID(): Returns dictionary of ids for each ngram and a dictionary of ngrams
    Input:  n    : Integer, Describes n for word/character n grams
            type : Integer, 0 for word Ngram, anything other integer for character Ngrams
    Output: word_dict: Returns dictionary of ids for each ngram
            ngrams: dictionary of word/character ngrams depending on the type
    """
    def generateNgramID(self, n, charGram=False):
        ind = 0
        ngram_dict = {}

        grams = self.genNgram(n) if charGram == False else self.genCharNgram(n)

        for seq in grams:
            if seq not in ngram_dict:
                ngram_dict[seq] = ind
                ind += 1

        self.gd = ngram_dict # gram dictionary
        self.ngrams = grams

        return ngram_dict, grams

    """
    getContentMatrix(): Returns the feature matrix for content features
    Input: cols : Integer, the number of columns in the feature matrix
           n    : Integer, Describes n for word n grams
    Output: Numpy Array of dimension(number of tweets, cols)
    """
    def getContentMatrix(self, n, tweetSet, param = 0):
        self.generateNgramID(n, False)

        fm = np.zeros((len(tweetSet), len(self.ngrams) + 8))
        temp_col = len(self.ngrams)

        for i in range(len(tweetSet)):
            ft = tweetSet[i]
            tweetNgram = ft.getNgram(n)
            for seq in tweetNgram:
                if seq in self.gd:
                    fm[i][self.gd[seq]] = float(tweetNgram[seq])
            if param > 0 :
                fm[i][temp_col] = ft.getAvgEmojis()
            if param > 1 :
                fm[i][temp_col + 1] = ft.getNumURL()
            if param > 2 :
                fm[i][temp_col + 2], fm[i][temp_col + 3] = ft.getNumTags()
            if param > 3:
                fm[i][temp_col + 4], fm[i][temp_col + 5], fm[i][temp_col + 6], dummy, dummy2 = ft.getPOSTaggedDistribution()
            if param > 4:
                fm[i][temp_col + 7] = ft.getNumTokens()
        return fm


    """
    getStylisticMatrix(): Returns the feature matrix for stylistic features
    Input: cols : Integer, the number of columns in the feature matrix
           n    : Integer, Describes n for character n grams
    Output: Numpy Array of dimension(number of tweets, cols)
    """
    def getStylisticMatrix(self, n, tweetSet, param = 0):

        self.generateNgramID(n, True) # Generate ngram ids
        fm = np.zeros( (len(tweetSet), len(self.ngrams) + 34)) # make np array

        temp_col = len(self.ngrams) # Keep track of temp_col

        # For each tweet
        # Fill in the matrix accordingly
        for i in range(len(tweetSet)):
            t = tweetSet[i]
            tweetCharGram = t.getCharNgram(n)

            for seq in tweetCharGram:
                if seq in self.gd:
                    fm[i][self.gd[seq]] = tweetCharGram[seq]
            if param > 0 :
                fm[i][temp_col] = t.getAvgNumPunct()
            if param > 1 :
                fm[i][temp_col + 1] = t.getAvgWordSize()
            if param > 2 :
                fm[i][temp_col + 2] = t.getVocabSize()
            if param > 3 :
                dummy, dummy1, dummy2, fm[i][temp_col + 3], fm[i][temp_col + 4] = t.getPOSTaggedDistribution()
            if param > 4 :
                fm[i][temp_col + 5] = t.getDigitFrequency()
            if param > 5 :
                fm[i][temp_col + 6] = t.getAvgHashTagLength()
            if param > 6 :
                fm[i][temp_col + 7] = t.getAvgCapitalizations()

            letters = t.getLetterFrequency()
            idx = 0
            if param > 7:
                for j in range(temp_col + 8, temp_col+32):

                    fm[i][j] = letters[idx]

                    idx += 1

        return fm

    """
    getStylisticMatrix(): Returns the feature matrix for stylistic features and content features concatenated together
    Input: fm : Numpy Array(float) of dimension(number of tweets, content features) of content features
           fv   : Numpy Array(float) of dimension(number of tweets, stylsitic features) of stylistic features
    Output: Numpy Array of dimension(number of tweets, content features + stylsitic features)
    """

    def getMergedMatrix(self, n, tweetSet, param):
        cm = self.getContentMatrix(1, tweetSet, 10)
        sm = self.getStylisticMatrix(3, tweetSet, 10)
        c = np.concatenate((cm, sm), 1)
        return c

    def getSplitData(self, n, param, X_train, X_test, Y_train, Y_test):
        # X_train, X_test, Y_train, Y_test = self.data.getRandomSplitData(.3)
        x_train = self.getMergedMatrix(n, X_train, param)
        x_test = self.getMergedMatrix(n, X_test, param)
        y_train = np.array(Y_train)
        y_test = np.array(Y_test)
        return x_train, y_train, x_test, y_test

    def getTrendMatrix(self):
        fm = np.zeros((len(self.tweets), 2))
        for i in range(len(self.tweets)):
            ft = Featurizer(self.tweets[i])
            fm[i][0] = ft.getLikes()
            fm[i][1] = ft.getRetweets()

if __name__ == "__main__":
    print("Running Vectorizer.py main")
    electionTweets = "./data/2016_US_election_tweets_100k.csv"
    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    d = Data(electionTweets, electionTrolls)
    f = Vectorizer(d)
    x_train, y_train, x_test, y_test = f.getSplitData(.3)
