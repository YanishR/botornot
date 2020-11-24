# Featurizer : Has a methods for parsing tweets to generate various stylistic and content based features
from nltk import word_tokenize

class Featurizer():
    # initializes object data
    tokens = []
    tweet = ""
    def __init__(self, tweet):
        self.tokens = word_tokenize(tweet)
        self.tweet = tweet
        pass


    """
    getNumHashTags(tweet): returns the number of hashtags in a tweet
    Input : Tweet object/ string
    Output: Number of hashtag tokens in the particular tweets
    """
    def getNumHashTags(self):
        count = 0
        for token in self.tokens:
            if token[0] == '#':
                count += 1
        return count





tweet = " Hi my name is Suvinay and this is my test tweet #test #final #random"
F = Featurizer(tweet)
print(F.getNumHashTags())
