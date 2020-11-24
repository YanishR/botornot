# Featurizer : Has a methods for parsing tweets to generate various stylistic and content based features
from nltk import word_tokenize

class Featurizer():
    # initializes object data
    def __init__(self):
        pass


    """
    getNumHashTags(tweet): returns the number of hashtags in a tweet
    Input : Tweet object/ string
    Output: Number of hashtag tokens in the particular tweets
    """
    def getNumHashTags(tweet):
        count = 0
        for token in word_tokenize(tweet):
            if token[0] == '#':
                count += 1
        return count


    tweet = " Hi my name is Suvinay and this is my test tweet #test #final #random"
    print(getNumHashTags(tweet))
