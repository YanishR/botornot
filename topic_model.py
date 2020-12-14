from dataparser import Data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import classification_report

# class to implement topic model using Latent Dirichlet Allocation
class TopicModel:
    # given an address to a dataset of legit tweets and another to fake tweets, constructs a TopicModel object
    def __init__(self, data = None, testing = False, training_data = None, training_label = None):
        self.legit, self.fake = [], []

        if testing == False:
            for tweet in data.getRealTweets():
                self.legit.append(tweet.getText())

            for tweet in data.getTrollTweets():
                self.fake.append(tweet.getText())

        elif testing:
            assert(len(training_data) == len(training_label))
            for i in range(len(training_data)):
                if training_label[i] == 0:
                    self.legit.append(training_data[i])
                else:
                    self.fake.append(training_data[i])

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
        return 0 if legit_similarity > fake_similarity else 1  # 0 represents legit, 1 represents fake

    # given a string tweet, returns its topic vector form
    def topic_vectorize(self, tweet):
        tweet_vector = self.cv_tweets.transform([tweet])

        topic_vector = self.lda_tweets.transform(tweet_vector)

        return topic_vector


if __name__ == "__main__":
    electionTweets = "./data/2016_US_election_tweets_100k.csv"

    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    data = Data(electionTweets, electionTrolls)

    x_train, x_test, y_train, y_test = data.getSplitDataDL(.3)

    tc = TopicModel(testing = True, training_data = x_train[:5000] , training_label = y_train[:5000])

    y_pred = []

    for x in x_test:
        y_pred.append(tc.classify(x))

    print(classification_report(y_test, y_pred))
