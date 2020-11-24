# Featurizer : Has a methods for parsing tweets to generate various stylistic and content based features
from nltk import word_tokenize

class Featurizer():
    # initializes object data
    tokens = []
    tweet = ""
    def __init__(self, tweet):
        self.tokens = tweet.split()
        self.tweet = tweet
        pass


    """
    getNumHashTags(tweet): returns the number of hashtags in a tweet
    Output: Number of hashtag tokens in the particular tweets
    """
    def getNumHashTags(self):
        count = 0
        for token in self.tokens:
            if token[0] == '#':
                count += 1
        return count

    """
    getNumTokens(tweet): returns the number of tokens in a tweet
    Output: Number of tokens in the particular tweet
    """
    def getNumTokens(self):
        return len(self.tokens)




tweet = " Hi my name is Suvinay and this is my test tweet #test #final #random"
F = Featurizer(tweet)
print(F.getNumTokens())
