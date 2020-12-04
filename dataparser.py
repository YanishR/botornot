# This module allows a user to retrieve tweets
# and read the content, time, likes,
# retweets, media and so on
import csv
from tweet import Tweet

from random import seed
from random import randint
import datetime

class Data:

    def __init__(self, realTweetsFileName, trollTweetsFileName):
        self.realTweets = readFile(realTweetsFileName, 11)
        self.trollTweets = readFile(trollTweetsFileName, 2)
            

    def getRealTweets(self):
        return self.realTweets

    def getTrollTweets(self):
        return self.trollTweets


    def getAllTweets(self):
        return self.trollTweets + self.realTweets

    def getSplitData(self):
        return self.realTweets[:int(len(self.realTweets)/4)],\
                self.trollTweets[:int(len(self.trollTweets)/4)],\
                self.realTweets[int(len(self.realTweets)/4):],\
                self.trollTweets[int(len(self.trollTweets)/4):]

    def getRandomizedSplitData(self, split=70):
        R_train, R_test = self.getRandomData(split, self.realTweets)
        T_train, T_test = self.getRandomData(split, self.trollTweets) 
        return R_train, T_train, R_test, T_test

    def getRandomData(self, split, tweetSet):
        train, test = [], []
        tweetsUsed = {}
        seed(datetime.datetime.now().second)

        while len(train) < split/100*len(tweetSet):
            value = randint(0, len(tweetSet) -1)
            if value not in tweetsUsed:
                tweetsUsed[value] = True
                train.append(tweetSet[value])

        for i in range(len(tweetSet)):
            if i not in tweetsUsed:
                test.append(tweetSet[i])

        return train, test 

def readFile(fileName, tPos):
    tweets = []
    with open(fileName) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) > 10:
                if row[tPos] != "" and row[tPos] != " ":
                    tweets.append(Tweet(row[tPos]))
    return tweets

if __name__ == "__main__":

    electionTweets = "./data/2016_US_election_tweets_100k.csv"

    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    tweets = Data(electionTweets, electionTrolls)

    print(tweets.getRandomizedSplitData())
