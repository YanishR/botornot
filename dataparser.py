# This module allows a user to retrieve tweets
# and read the content, time, likes,
# retweets, media and so on
import csv

class Data:

    def __init__(self, realTweetsFileName,trollTweetsFileName):

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

    electionTweets = "./data/2016_US_election_tweets_100k.csv"

    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    tweets = Data(electionTrolls, electionTweets)

    tweets.getSplitData()
