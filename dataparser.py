# This module allows a user to retrieve tweets
# and read the content, time, likes,
# retweets, media and so on
import csv

class Data:

    def __init__(self, trollTweetsFileName, realTweetsFileName):

        self.trollTweets = readFile(trollTweetsFileName, 2)
        self.realTweets = readFile(realTweetsFileName, 11)
            


    def getTrollTweets(self):
        return self.trollTweets

    def getRealTweets(self):
        return self.realTweets

    def getAllTweets(self):
        return self.trollTweets + self.realTweets

    def getSplitData(self):
        return self.realTweets[:int(len(self.realTweets)/4)],\
                self.trollTweets[:int(len(self.trollTweets)/4)],\
                self.realTweets[int(len(self.realTweets)/4):],\
                self.trollTweets[int(len(self.trollTweets)/4):]

def readFile(fileName, tPos):
    tweets = []
    with open(fileName) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) > 10:
                if row[tPos] != "" and row[tPos] != " ":
                    tweets.append(row[tPos])
    return tweets

if __name__ == "__main__":
    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    electionTweets = "./data/2016_US_election_tweets_100k.csv"


    tweets = Data(electionTrolls, electionTweets)
    tweets.getSplitData()
