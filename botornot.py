from dataparser import Data
from Vectorizer import Vectorizer

if __name__ == "__main__":
    print("Running botornot")

    # Declare files
    electionTweets = "./data/2016_US_election_tweets_100k.csv"
    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    v = Vectorizer(electionTweets, electionTrolls)
    v.runSVM()
    
    accs = []
    for i in range(0, 500, 100):
        p = v.runKBestFeatures(i, 3, content=True)

        accs.append(p["accuracy"])
    print(accs)
