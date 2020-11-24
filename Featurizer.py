# Featurizer : Has a methods for parsing tweets to generate various stylistic and content based features
from nltk.tag import pos_tag
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
    Output: Float number of emojis normalized by tokens in tweet
    Discuss: Tweets need to be preprocessed so as to be word tokenized rather than split, to be done in the future
    """
    def getAvgEmojis(self):
        count = 0
        for token in self.tokens:
            if token in UNICODE_EMOJI:
                count += 1
        return count/self.getNumTokens()


    """
    getDigitFrequency(): Returns the average number of  digits in a tweet normalized by the number of tokens in the tweet
    Output: Float number of digits normalized by number of tokens in tweet
    """
    def getDigitFrequency(self):
        count = 0
        for char in self.tweet:
            if char.isdigit():
                count += 1
        return count/self.getNumTokens()


    """
    getPOSTaggedDistribution(): Returns the number of nouns, adjectives, adverbs, verbs, conjunctions in the tweet after POS tagging
    Output: noun_count : noun_count: Number of nouns in the tweets
                         adj_count:  Number of adjectives in the Tweets
                         adv_count: Number of adverbs in the Tweets
                         verb_count: Number of verbs in the Tweets
                         conj_count: Number of conjunctions in the Tweets
    """
    def getPOSTaggedDistribution(self):
        tagged = pos_tag(self.tokens)
        noun_count, adj_count, adv_count, verb_count, conj_count = 0, 0, 0, 0, 0
        for tuple in tagged:
            if tuple[1][0] == 'N' and tuple[1][1] == 'N':
                noun_count += 1
            elif tuple[1][0] == 'V' and tuple[1][1] == 'B':
                verb_count += 1
            elif tuple[1][0] == 'J' and tuple[1][1] == 'J':
                adj_count += 1
            elif tuple[1][0] == 'C' and tuple[1][1] == 'C':
                conj_count += 1
            elif tuple[1][0] == 'R' and tuple[1][1] == 'B':
                adv_count += 1
        return noun_count, adj_count, adv_count, verb_count, conj_count


    """
    getDigitFrequency(): Returns the average size of hashtagged strings in tweet
    Output: Float length of hashtag words normalized by number of hashtags
    """
    def getAvgHashTagLength(self):
        tag_count = 0
        size_count = 0
        for token in self.tokens:
            if token[0] == '#':
                tag_count += 1
                size_count += len(token)
        return size_count/tag_count



















tweet = " Hi, my name is Suvinay. This is www.hotmail.com my e-mail service. #Hypochondriac #ThisIsAVeryLongHashTag #ABC"
F = Featurizer(tweet)
print(F.getAvgHashTagLength())
