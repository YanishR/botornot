# This module allows a user to retrieve tweets
# and read the content, time, likes,
# retweets, media and so on
import csv
from tweet import Tweet

# Get split method
from sklearn.model_selection import train_test_split

class Data:

    # Init method
    def __init__(self, realTweetsFileName, trollTweetsFileName, realTweetSize=5000, trollTweetSize=1000):
        # Read in real tweets and troll tweets from given files
        # 11, 2 denote the index of text in csv
        self.realTweets = readFile(realTweetsFileName, 11)
        self.trollTweets = readFile(trollTweetsFileName, 2)

        # Get size of tweet sets
        realSize = realTweetSize if 0 < realTweetSize < len(self.realTweets) else len(self.realTweets)
        trollSize = trollTweetSize if 0 < trollTweetSize < len(self.trollTweets) else len(self.trollTweets)

        # Reduce data sizes
        self.realTweets = self.realTweets[:realSize]
        self.trollTweets = self.trollTweets[:trollSize]
        self.tweets = self.realTweets + self.trollTweets

    # Get All real tweets
    def getRealTweets(self):
        return self.realTweets

    # Get all troll tweets
    def getTrollTweets(self):
        return self.trollTweets

    # Get all tweets
    def getAllTweets(self):
        return self.tweets

    # Split data randomly given a test set size
    def getRandomSplitData(self, testSize):
        # Make array of classifications
        y = [0] * len(self.realTweets)
        y += [1] * len(self.trollTweets)
        # Using train_test_split, get the split
        X_train, X_test, y_train, y_test =\
                train_test_split(self.tweets, y, test_size = testSize, shuffle=True)

        # Return 
        return X_train, X_test, y_train, y_test

    # Split data for Deep Learning
    def getSplitDataDL(self, testSize):
        X_train, X_test, y_train, y_test = self.getRandomSplitData(testSize)

        x_train, x_test = [t.getText() for t in X_train], [t.getText() for t in X_test]

        return x_train, x_test, y_train, y_test

# Reads in a csv file with csv methods
def readFile(fileName, tPos):
    # Create list of tweets
    tweets = []
    with open(fileName) as csvfile:
        # Initialize reader
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) > 10:
                if row[tPos] != "" and row[tPos] != " ":
                    # Append tweets making a Tweet Object
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
