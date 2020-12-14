from dataparser import Data
from Vectorizer import Vectorizer
from topic_model import TopicModel

if __name__ == "__main__":
    print("Running botornot")

    # Declare files
    electionTweets = "./data/2016_US_election_tweets_100k.csv"
    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    v = Vectorizer(electionTweets, electionTrolls)

    # Run SVM with default features
    # Features such as content matrix and kernel type
    # can be changed
    # See README for more information
    # Results will be printed
    v.runSVM()

    # Run K best features with
    # specific features
    # k = 1000
    # contentMatrix contains all features (1 to 5)
    # n for word n-gram: ¡
    # stylistic matrix contains all features (1 to 7)
    # n for char n-gram:3
    f, p = v.runKBestFeatures(1000, contentMatrix=True, contentMatrixFeatures=[i for i in range(5)], wordN=1,\
            stylisticMatrix=True, stylisticMatrixFeatures = [i for i in range(7)], charN=3, testSize=.3)


    """
    Uncomment for k best merged matrix

    for i in [100, 500, 1000]:
        print("k = " +  str(i))
        f, p = v.runKBestFeatures(1000, contentMatrix=True, contentMatrixFeatures=[i for i in range(5)], wordN=1,\
                stylisticMatrix=True, stylisticMatrixFeatures = [i for i in range(7)], charN=3, testSize=.3)

        #Print features if desired

        #for feature in f:
            #print(feature)
    """

    """
    Uncomment for LDA based classification

    data = Data(electionTweets, electionTrolls)

    x_train, x_test, y_train, y_test = data.getSplitDataDL(.3)

    tc = TopicModel(testing = True, training_data = x_train[:5000] , training_label = y_train[:5000])

    y_pred = []

    for x in x_test:
        y_pred.append(tc.classify(x))

    print(classification_report(y_test, y_pred))
    """
