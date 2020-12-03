from tweet.py import Tweet
import numpy as np

"""
Vectorizer:
    Class that transforms two lists of troll Tweets and real Tweets
"""
class Vectorizer():

    def __init__(self, realTweets, trollTweets):
        self.trollTweets = []
        self.realTweets = []

        for tweet in trollTweets:
            self.trollTweets.append(Tweet(tweet))
        for tweet in realTweets:
            self.realTweets.append(Tweet(tweet))


    """
    generateNgram(): Returns dictionary of ids for each ngram and a dictionary of ngrams
    Input:  n    : Integer, Describes n for word/character n grams
            type : Integer, 0 for word Ngram, anything other integer for character Ngrams
    Output: word_dict: Returns dictionary of ids for each ngram
            ngrams: dictionary of word/character ngrams depending on the type
    """
    def genNgram(self, n):
        g = {}
        for tweet in self.tweets:
            for ngram in tweet.getNgram(n):
                if ngram in g:
                    g[ngram] += 1
                else:
                    g[ngram] = 1
        return g


    """
    generateNCharNgram(): Returns dictionary of ids for each ngram and a dictionary of ngrams
    Input:  n    : Integer, Describes n for word/character n grams
            type : Integer, 0 for word Ngram, anything other integer for character Ngrams
    Output: word_dict: Returns dictionary of ids for each ngram
            ngrams: dictionary of word/character ngrams depending on the type
    """
    def getCharNgram(self, n):
        g = {}
        for tweet in self.tweets:
            for gram in tweet.getCharNgram(n):
                if gram in g:
                    g[gram] += 1
                else:
                    g[gram] = 1
        return g


    """
    generateNgramID(): Returns dictionary of ids for each ngram and a dictionary of ngrams
    Input:  n    : Integer, Describes n for word/character n grams
            type : Integer, 0 for word Ngram, anything other integer for character Ngrams
    Output: word_dict: Returns dictionary of ids for each ngram
            ngrams: dictionary of word/character ngrams depending on the type
    """
    def generateNgramID(self, n, choice):
        ind = 0

        ngrams = getNgram(n) if choice == 0 else getCharNgram(n)

        word_dict = {}

        for seq in ngrams:
            if seq not in word_dict:
                word_dict[seq] = ind
                ind += 1

        self.wd = word_dict
        return word_dict, ngrams


    """
    getContentMatrix(): Returns the feature matrix for content features
    Input: cols : Integer, the number of columns in the feature matrix
           n    : Integer, Describes n for word n grams
    Output: Numpy Array of dimension(number of tweets, cols)
    """
    def getContentMatrix(self, n, cols):
        self.generateNgramID(n, 0)
        fm = np.zeros((len(self.tweets)), cols)

        for i in range(len(self.realTweets)):
            temp_col = 0

            for seq in self.realTweets[i].getNgram(n):
                fm[i][temp_col]] = float(self.wd[seq])
                temp_col += 1

            fm[i][temp_col] = ft.getAvgEmojis()
            fm[i][temp_col + 1] = ft.getNumURL()
            fm[i][temp_col + 2], fm[i][temp_col + 3] = ft.getNumTags()
            fm[i][temp_col + 4], fm[i][temp_col + 5], fm[i][temp_col + 6], dummy, dummy2 = ft.getPOSTaggedDistribution()
            fm[i][temp_col + 7] = ft.getNumTokens()

        return fm


    """
    getStylisticMatrix(): Returns the feature matrix for stylistic features
    Input: cols : Integer, the number of columns in the feature matrix
           n    : Integer, Describes n for character n grams
    Output: Numpy Array of dimension(number of tweets, cols)
    """
    def getStylisticMatrix(self, cols, n):
        self.generateNgramID(n, 1)
        fm = np.zeros( (len(self.tweets)), cols)
        self.generateNgramID(n, 1)

        for i in range(len(self.realTweets)):
            temp_col = 0

            for seq in self.realTweets[i].getCharNgram(n):
                fm[i][temp_col] = float(self.wd[seq])
                temp_col += 1

            fm[i][temp_col] = ft.getAvgNumPunct()
            fm[i][temp_col + 1] = ft.getAvgWordSize()
            fm[i][temp_col + 2] = ft.getVocabSize()
            dummy, dummy1, dummy2, fm[i][temp_col + 3], fm[i][temp_col + 4] = ft.getPOSTaggedDistribution()
            fm[i][temp_col + 5] = ft.getDigitFrequency()
            fm[i][temp_col + 6] = ft.getAvgHashTagLength()
            fm[i][temp_col + 7] = ft.getAvgCapitalizations()

            letters = ft.getLetterFrequency()
            idx = 0
            for j in range(temp_col+8, temp_col+33):
                fm[i][j] = letters[idx]
                idx += 1

        return fm

    """
    getStylisticMatrix(): Returns the feature matrix for stylistic features and content features concatenated together
    Input: fm : Numpy Array(float) of dimension(number of tweets, content features) of content features
           fv   : Numpy Array(float) of dimension(number of tweets, stylsitic features) of stylistic features
    Output: Numpy Array of dimension(number of tweets, content features + stylsitic features)
    """
    def getMergedMatrix(self, fm, fv):
        return np.concatenate((fm,fv), 1)

if __name__ == "__main__":
    print("Running main")
    electionTweets = "./data/2016_US_election_tweets_100k.csv"
    electionTrolls = "./data/russian-troll-tweets/IRAhandle_tweets_1.csv"

    d = Data(electionTweets, electionTrolls)
    f = Vectorizer(d.getRealTweets(), d.getTrollTweets())

