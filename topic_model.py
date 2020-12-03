from dataparser import Data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# class to implement topic model using Latent Dirichlet Allocation
class TopicModel:
    # given an address to a dataset of legit tweets and another to fake tweets, constructs a TopicModel object
    def __init__(self, legit, fake):
        electionTweets = Data(legit, fake)


        self.legit = electionTweets.getRealTweets()[:1000]

        self.fake = electionTweets.getTrollTweets()[:1000]

        self.tweets = self.legit + self.fake

        self.cv_tweets = CountVectorizer(max_df = 0.95, min_df = 2, stop_words = 'english')
        self.cv_tweets.fit(self.tweets)
        self.df_tweets = self.cv_tweets.transform(self.tweets)


        self.lda_tweets = LatentDirichletAllocation(n_components = 20, learning_method = 'online', random_state = 42)

        self.doc_top = self.lda_tweets.fit_transform(self.df_tweets)


    # given a string tweet, classifies it as either legit or fake using topic model
    def classify(self, tweet):
        tweet_vector = self.cv_tweets.transform([tweet])

        topic_vector = self.lda_tweets.transform(tweet_vector)

        max_legit_vec = max(   self.doc_top[:len(self.legit)], key = lambda t : cosine_similarity(  topic_vector, np.array(t).reshape(1,-1)  )[0][0]   )

        max_fake_vec = max(   self.doc_top[len(self.legit):], key = lambda t : cosine_similarity(  topic_vector, np.array(t).reshape(1,-1)  )[0][0]   )

        legit_similarity = cosine_similarity(   topic_vector, max_legit_vec.reshape(1,-1)  )[0][0]
        fake_similarity = cosine_similarity(   topic_vector, max_fake_vec.reshape(1,-1)  )[0][0]
        #
        return "legit" if legit_similarity > fake_similarity else "fake"

    # given a string tweet, returns its topic vector form
    def topic_vectorize(self, tweet):
        tweet_vector = self.cv_tweets.transform([tweet])

        topic_vector = self.lda_tweets.transform(tweet_vector)

        return topic_vector


if __name__ == "__main__":
    electionTweets = "./data/2016_US_election_tweets_100k.csv"

    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    tc = TopicModel("./data/2016_US_election_tweets_100k.csv", "./data/IRAhandle_tweets_1.csv")

    print(tc.classify("i'm going to vote for hillary clinton, thanks!"))
    print(tc.topic_vectorize("please vote for donald trump"))
    # print("LEGITIMATE TWEETS:\n\n")
    # for index, topic in enumerate(lda_tweets.components_):
    #     print(f'Top 15 words for Topic #{index}')
    #     print([cv_tweets.get_feature_names()[i] for i in topic.argsort()[-15:]])
    #     print('\n')
    #
    # print("TROLL TWEETS:\n\n")
    # for index, topic in enumerate(lda_trolls.components_):
    #     print(f'Top 15 words for Topic #{index}')
    #     print([cv_trolls.get_feature_names()[i] for i in topic.argsort()[-15:]])
    #     print('\n')
