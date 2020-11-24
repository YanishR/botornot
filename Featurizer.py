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
    getNumTags(): returns the number of hashtags and the number of @ mentions
    Output: Integer number of hashtag tokens in the particular tweet
            Integer number of @ mentions in the particular tweet
    Discuss: If every character is checked repeated useless hashtags are counted,
             If every character at index 0 for every token is check, hashtags not spaced are not counted as multiple
    """
    def getNumTags(self):
        hash_count = 0
        rate_count = 0
        for char in self.tweet:
            if char == '#':
                hash_count += 1
            if char == '@'
                rate_count += 1
        return hash_count, rate_count


    """
    getNumTokens(): returns the number of tokens in a tweet
    Output: Number of tokens in the particular tweet
    """
    def getNumTokens(self):
        return len(self.tokens)


    """
    getAvgWordSize(): returns the average word size (character length) in a tweet
    Output: average of type float word size (character length) in a tweet
    """
    def getAvgWordSize(self):
        count = 0
        for token in self.tokens:
            count += len(token)
        return count/self.getNumTokens()


    """
    getNumPunct(): returns the number of punctuations used in a tweet normalized by token length
    Output: float average number of punctuations
    """
    def getAvgNumPunct(self):
        count = 0
        punct_set = set(['.',',',';',':','?','-','!',"'",'"','[',']','(', ')', '{', '}'])
        for char in self.tweet:
            if char in punct_set:
                count += 1
        return count/self.getNumTokens()









tweet = " Hi, my name is Suvinay. This is my test tweet #test #final #random"
F = Featurizer(tweet)
print(F.getAvgNumPunct())
