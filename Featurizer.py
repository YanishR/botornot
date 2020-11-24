# Featurizer : Has a methods for parsing tweets to generate various stylistic and content based features
from nltk import word_tokenize
import re
from emoji import UNICODE_EMOJI # NOTE: pip3 install emoji


class Featurizer():
    # initializes object data
    tokens = []
    tweet = ""
    def __init__(self, tweet):
        self.tokens = tweet.split()
        self.tweet = tweet
        pass


    """
    getNumTags(): Returns the number of hashtags and the number of @ mentions
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
            if char == '@':
                rate_count += 1
        return hash_count, rate_count


    """
    getNumTokens(): Returns the number of tokens in a tweet
    Output: Number of tokens in the particular tweet
    """
    def getNumTokens(self):
        return len(self.tokens)


    """
    getAvgWordSize(): Returns the average word size (character length) in a tweet
    Output: Average of type float word size (character length) in a tweet (string)
    """
    def getAvgWordSize(self):
        count = 0
        for token in self.tokens:
            count += len(token)
        return count/self.getNumTokens()


    """
    getNumPunct(): Returns the number of punctuations used in a tweet normalized by token length
    Output: Float average number of punctuations in a string tweet
    """
    def getAvgNumPunct(self):
        count = 0
        punct_set = set(['.',',',';',':','?','-','!',"'",'"','[',']','(', ')', '{', '}'])
        for char in self.tweet:
            if char in punct_set:
                count += 1
        return count/self.getNumTokens()

    """
    getNumURL(): Returns the number of URLs in a tweet
    Output: Integer number of URLs in a given string tweet
    """
    def getNumURL(self):
        #regex citation: GeeksForGeeks
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        url = re.findall(regex,tweet)
        return len([x[0] for x in url])


    """
    getNumTokens(): Returns the number of tokens in a tweet
    Output: Integer number of tokens in the particular tweet
    Discuss: Tweets need to be preprocessed so as to be word tokenized rather than split, to be done in the future
    """
    def getVocabSize(self):
        return len(set([word.lower() for word in self.tokens]))


    """
    getAvgCapitalizations(): Returns the average number of capitalized tokens in a tweet normalized by the number of tokens in the tweet
    Output: Float number of normalized tokens with a capitalized first character in the particular tweet
    Discuss: Tweets need to be preprocessed so as to be word tokenized rather than split, to be done in the future
    """
    def getAvgCapitalizations(self):
        count = 0
        for token in self.tokens:
            if token[0].isupper():
                count += 1
        return count/self.getNumTokens()


    """
    getAvgEmojis(): Returns the average number of  emoji tokens in a tweet normalized by the number of tokens in the tweet
    Output: Float number of normalized tokens with emojis fir
    Discuss: Tweets need to be preprocessed so as to be word tokenized rather than split, to be done in the future
    """
    def getAvgEmojis(self):
        count = 0
        for token in self.tokens:
            if token in UNICODE_EMOJI:
                count += 1
        return count/self.getNumTokens()














tweet = " Hi, my name is Suvinay. This is www.hotmail.com my test tweet #test #final #random www.google.com https://www.facebook.com"
F = Featurizer(tweet)
print(F.getAvgCapitalizations())
