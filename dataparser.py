# This module allows a user to retrieve tweets
# and read the content, time, likes,
# retweets, media and so on
import csv
from tweet import Tweet

from sklearn.model_selection import train_test_split
from random import seed
from random import randint

import datetime

class Data:

    def __init__(self, realTweetsFileName, trollTweetsFileName):
        self.realTweets = readFile(realTweetsFileName, 11)
        self.trollTweets = readFile(trollTweetsFileName, 2)

        m = min(len(self.realTweets), len(self.trollTweets))
        m = 10000

        self.realTweets = self.realTweets[:m]
        self.trollTweets = self.trollTweets[:m]
        self.tweets = self.realTweets + self.trollTweets

    def getRealTweets(self):
        return self.realTweets

    def getTrollTweets(self):
        return self.trollTweets

    def getAllTweets(self):
        return self.tweets

    def getRandomSplitData(self, testSize):
        y = [0] * len(self.realTweets)
        y += [1] * len(self.trollTweets)
        X_train, X_test, y_train, y_test =\
                train_test_split(self.tweets, y, test_size = testSize, shuffle=True)

        return X_train, X_test, y_train, y_test

    def getSplitDataDL(self, testSize):
        X_train, X_test, y_train, y_test = self.getRandomSplitData(testSize)

        x_train, x_test = [t.getText() for t in X_train], [t.getText() for t in X_test]

        return x_train, x_test, y_train, y_test

def readFile(fileName, tPos):
    tweets = []
    with open(fileName) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) > 10:
                if row[tPos] != "" and row[tPos] != " ":
                    tweets.append(Tweet(row[tPos]))
    return tweets

"""
Main for testing purposes

"""
if __name__ == "__main__":
    print("Running dataparser.py main")

    electionTweets = "./data/2016_US_election_tweets_100k.csv"
    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    tweets = Data(electionTweets, electionTrolls)
    tweets.getSplitDataDL(.7)
    #print(tweets.getRandomizedSplitData())
