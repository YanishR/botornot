# This module allows a user to retrieve tweets
# and read the content, time, likes,
# retweets, media and so on
import csv

class Data:

    def __init__(self, fileName, trolls=False):

        self.tweets = []
        
        contentPos = 11
        if trolls: 
            contentPos = 2

        with open(fileName) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row) > 10:
                    if row[contentPos] != "" and row[contentPos] != " ":
                        self.tweets.append(row[contentPos])

            

    def getTweets(self):
        return self.tweets

    def getTweet(self, i):
        return self.tweets[i]
        
if __name__ == "__main__":
    electionTweets = "./data/2016_US_election_tweets_100k.csv"

    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    electionTweets = Data(electionTweets)
    print(len(electionTweets.getTweets()))

    electionTrolls = Data(electionTrolls, True)

    #for tweet in electionTrolls.getTweets():
        #print(tweet)
