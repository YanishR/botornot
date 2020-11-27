# Featurizer : Has a methods for parsing tweets to generate various stylistic and content based features
import re
import numpy as np
from nltk.tag import pos_tag

from emoji import UNICODE_EMOJI # NOTE: pip3 install emoji

# Get ngram computation method
from nltk import ngrams


class Featurizer():
    # Need to improve object oriented design further in the future
    # initializes object data

    def __init__(self, tweet):
        self.tokens = tweet.split()
        self.tweet = tweet
        self.letterFreq = None


    """
    getNumTags(): Returns the number of hashtags and the number of @ mentions
    Output: Integer number of hashtag tokens in the particular tweet
            Integer number of @ mentions in the particular tweet
    Discuss: If every character is checked repeated useless hashtags are counted,
             If every character at index 0 for every token is check, hashtags not spaced are not counted as multiple
    """
    def getNumTags(self):
        if not self.hash_count and self.rate_count:

            self.hash_count = 0
            self.rate_count = 0

            for char in self.tweet:
                if char == '#':
                    self.hash_count += 1
                if char == '@':
                    self.rate_count += 1

        return self.hash_count, self.rate_count


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
        if not self.avgWordSize:
            count = 0
            for token in self.tokens:
                count += len(token)
            self.avgWordSize = count/self.getNumTokens()

        return self.avgWordSize


    """
    getNumPunct(): Returns the number of punctuations used in a tweet normalized by token length
    Output: Float average number of punctuations in a string tweet
    """
    def getAvgNumPunct(self):
        if not self.avgNumPunct:
            count = 0
            punct_set = set(['.',',',';',':','?','-','!',"'",'"','[',']','(', ')', '{', '}'])
            for char in self.tweet:
                if char in punct_set:
                    count += 1
                
            self.avgNumPunct = count/self.getNumTokens()
        return self.avgNumPunct

    """
    getNumURL(): Returns the number of URLs in a tweet
    Output: Integer number of URLs in a given string tweet
    """
    def getNumURL(self):
        if not self.numURL:
            #regex citation: GeeksForGeeks
            regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
            url = re.findall(regex,self.tweet)
            self.numURL = len([x[0] for x in url])

        return self.numURL

    """
    getNumTokens(): Returns the number of tokens in a tweet
    Output: Integer number of tokens in the particular tweet
    Discuss: Tweets need to be preprocessed so as to be word tokenized rather than split, to be done in the future
    """
    def getVocabSize(self):
        if not self.vocabSize:
            self.vocabSize = len(set([word.lower() for word in self.tokens]))
        return self.vocabSize

    """
    getAvgCapitalizations(): Returns the average number of capitalized tokens in a tweet normalized by the number of tokens in the tweet
    Output: Float number of normalized tokens with a capitalized first character in the particular tweet
    Discuss: Tweets need to be preprocessed so as to be word tokenized rather than split, to be done in the future
    """
    def getAvgCapitalizations(self):
        if not self.avgCaps:
            count = 0
            for token in self.tokens:
                if token[0].isupper():
                    count += 1
            self.avgCaps = count/self.getNumTokens()
        return self.avgCaps


    """
    getAvgEmojis(): Returns the average number of  emoji tokens in a tweet normalized by the number of tokens in the tweet
    Output: Float number of emojis normalized by tokens in tweet
    Discuss: Tweets need to be preprocessed so as to be word tokenized rather than split, to be done in the future
    """
    def getAvgEmojis(self):
        if not self.avgEmojis:
            count = 0
            for token in self.tokens:
                if token in UNICODE_EMOJI:
                    count += 1
            self.avgEmojis = count/self.getNumTokens()
        return self.avgEmojis


    """
    getDigitFrequency(): Returns the average number of  digits in a tweet normalized by the number of tokens in the tweet
    Output: Float number of digits normalized by number of tokens in tweet
    """
    def getDigitFrequency(self):
        if not self.digitFrequency:
            count = 0
            for char in self.tweet:
                if char.isdigit():
                    count += 1
            self.digitFrequency = count/self.getNumTokens()
        return self.digitFrequency


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
        if not self.avgHashTagLength:
            tagCount, sizeCount = 0, 0
            for token in self.tokens:
                if token[0] == '#':
                    tagCount += 1
                    sizeCount += len(token)
            self.avgHashTagLength = sizeCount/tagCount
        return self.avgHashTagLength


    """
    getLetterFrequency(): Returns the frequency of each letter (Case insensitive) normalized by the number of tokens in the tweet
    Output: Array of frequencies for each letter, where each index corresponds to its number in the alphabet
    """
    def getLetterFrequency(self):
        if not self.letterFreq:
            alphabet = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',\
                    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',\
                    's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            self.letterFreq = [self.tweet.lower().count(alphabet[i]) for i in range(0,26)]
        return self.letterFreq

    def getNgram(self, n):
        g = {}
        for seq in ngrams(self.tokens, n):
            s = ""
            for w in seq:
                s += w + " "
            s = s[:-1]
            if s in g:
                g[s] += 1
            else:
                g[s] = 1
        return g 

    def getCharNgram(self, n):
        g = {}
        print(ngrams(self.tweet, n))
        for s in ngrams(self.tweet, n):
            print(s)
        for seq in ngrams(self.tweet, n):
            s = ""
            for c in seq:
                s += c
            if s in g:
                g[s] += 1
            else:
                g[s] = 1
        return g 

if __name__ == "__main__":

    tweet = " aaa bb cccc dd e y zz"
    F = Featurizer(tweet)
    print(F.getNgram(2)) 
    print(F.getCharNgram(2))
    print(F.getLetterFrequency())
