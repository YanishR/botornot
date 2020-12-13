# Local module imports
from dataparser import Data

# numpy import
import numpy as np

#sci kit imports
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier

"""
Vectorizer:
    Class that takes in data of real tweets and troll tweets,
    transforms the data into useful feature matrices,
    and runs experiments:
    - SVM with different parameter 
    - k best features with different parameters
"""
class Vectorizer():
    """
    __init__(): Initializes our vectorizer with a real tweet and troll tweet file
    Input:  realTweetFile: specifying path of real tweets
            trollTweetFile: self explanatory 
            realTweetSize: self explanatory
            trollTweetSize: self explanatory
    """

    def __init__(self, realTweetFile, trollTweetFile, realTweetSize=5000, trollTweetSize=1000):
        self.data = Data(realTweetFile, trollTweetFile, realTweetSize, trollTweetSize)
    
    """
    runSVM(): Performs SVM classification and prints results
    Input:  contentMatrix: Boolean of whether we want the content matrix as features
            contentMatrixFeatures: if contentMatrix, specify features to be used
            wordN: if contentMatrix, n for word n-grams
            stylisticMatrix: As before
            stylisticMatrixFeatures: As before
            charN: As before
            testSize: How big we want our test set compare to entire data set
            r: r for svc 
            hyper_param: for svc
            kernel_type: 0 is linear, 1 is poly
            degree: degree of function
    Output: performance: dictionary of performance metrics
    """
    def runSVM(self, contentMatrix=True, contentMatrixFeatures=[], wordN=1,\
            stylisticMatrix=True, stylisticMatrixFeatures=[], charN=3, testSize=.3,\
            r=1, hyper_param=0.1, kernel_type=0, degree=1):
        
        matrices = {} # Dictionary to pass information to getSplitData

        # If content matrix, build appropriate data
        if contentMatrix:
            matrices['content'] = [wordN, contentMatrixFeatures]

        # Same for stylistic
        if stylisticMatrix:
            matrices['stylistic'] = [charN, stylisticMatrixFeatures]

        # Get split data with matrices and desired test size
        X_train, Y_train, X_test, Y_test = self.getSplitData(matrices, testSize)

        # Declare SVC classifier
        svc = SVC(C = hyper_param, kernel = 'linear', degree = degree, class_weight = 'balanced')\
            if kernel_type == 0\
            else SVC(C = hyper_param, kernel = 'poly', degree = degree, class_weight = 'balanced', coef0 = r)

        # Fit the data
        svc.fit(X_train, Y_train)
        # And obtain Y_predicted
        Y_predicted = svc.predict(X_test)
        # Has to be done for auroc
        Y_predicted_auroc = svc.decision_function(X_test)
        # Get performance metrics
        performance = self.getPerformance(Y_test, Y_predicted, Y_predicted_auroc)

        # Print results
        print("Performing experiment")
        print("---------------------------------------")
        if kernel_type == 0:
            print("Linear kernel with: ")
        else:
            print("Quadratic kernel with: ")

        print("r: " + str(r) + ", hyper parameter: " + str(hyper_param))

        print("Content Matrix: " + str(contentMatrix))
        if contentMatrix:
            print("Word ngram n: " + str(wordN))
            print("Content Matrix Features: " + str(contentMatrixFeatures))

        print("Stylistic Matrix: " + str(stylisticMatrix))
        if stylisticMatrix:
            print("Char ngram n: " + str(charN))
            print("Stylistic Matrix Features" + str(stylisticMatrixFeatures))

        print("test set size: " + str(testSize))
        print("-----------------------------------------")
        print("Performance")
        for metric in performance:
            print(metric + " : " + str(performance[metric]))

        return performance
    
    """
    runKBestFeatures(): Performs select K best features
    Input:  k: for k best features
            n: for word or char gram n
            content: boolean denoting whether content matrix is to be used
            or stylistic matrix
            features: features for the matrix to be used
            kernel_type, r, degree and hyper_param as before
    Output: features: list of k best features
            performance: dictionary of performance metrics
    """
    def runKBestFeatures(self, k, n, content=True, features=[], kernel_type=0, r=1,degree=1, hyper_param=1):

        # Build matrix for info
        m = {}
        
        if content:
            m['content'] = [n, features]
        else:
            m['stylistic'] = [n, features]

        # Get split data from m
        X_train, Y_train, X_test, Y_test = self.getSplitData(m)

        # Find best k features
        X_new = SelectKBest(chi2, k=k).fit(X_train, Y_train) # Get the best scores
        
        # Create dictionary to search through
        scores = {}         # Add best scores to s
        for i in range(len(X_new.scores_)):
            if not np.isnan(X_new.scores_[i]):
                scores[X_new.scores_[i]] = i

        # Create list for best k features
        features = []
        ids = self.generateNgramID(n) if content else self.generateNgramID(n, True)

        # Get top k features from s
        for s in sorted(scores.items(), reverse=True)[:k]:
            # boolean to see if it is an ngram
            found = False
            for seq in ids:
                if s[1] == ids[seq]:
                    features.append(seq)
                    found = True
            # If not found among ngrams
            if found == False:
                features.append(s[1] - len(ids))

        # Fit with traning data
        kBest = SelectKBest(chi2, k=100).fit(X_train, Y_train)

        # transform training and test X
        X_train_new = kBest.transform(X_train)
        X_test_new = kBest.transform(X_test)

        # Declare SVC
        svc = SVC(C = hyper_param, kernel = 'linear', degree = 1, class_weight = 'balanced')\
            if kernel_type == 0\
            else SVC(C = hyper_param, kernel = 'poly', degree = 2, class_weight = 'balanced', coef0 = r)

        # Fit data
        svc.fit(X_train_new, Y_train)
        Y_predicted = svc.predict(X_test_new)
        Y_predicted_auroc = svc.decision_function(X_test_new)
        performance = self.getPerformance(Y_test, Y_predicted, Y_predicted_auroc)

        # Print results
        print("Performing experiment")
        print("---------------------------------------")
        if kernel_type == 0:
            print("Linear kernel with: ")
        else:
            print("Quadratic kernel with: ")

        print("r: " + str(r) + ", hyper parameter: " + str(hyper_param))

        print("Content Matrix: " + str(content))
        if content:
            print("Word ngram n: " + str(n))
            print("Content Matrix Features: " + str(features))

        print("Stylistic Matrix: " + str(content==False))
        if content==False:
            print("Char ngram n: " + str(n))
            print("Stylistic Matrix Features" + str(features))

        print("test set size: " + str(.3))
        print("-----------------------------------------")
        print("Performance")
        for metric in performance:
            print(metric + " : " + str(performance[metric]))
        print("-----------------------------------------")

        return features, performance 

    """
    getSplitData(): Returns split data given dictionary of desired features
    for matrices
    Input:  matrices: dictionary informing about desired features
            testSize: size of test set compared to entire data set
    Output: X_train, Y_train, X_test, Y_test 
    """
    
    def getSplitData(self, matrices={'content':[1, []], 'stylistic':[3, []]}, testSize=0.3):
        # Get split data from self.data with testSize
        X_train, X_test, Y_train, Y_test = self.data.getRandomSplitData(testSize)
        
        # Declare variable for content train and test matrices
        cmTr, cmTe = None, None

        if 'content' in matrices:
            cmTr = self.getContentMatrix(X_train, matrices['content'][0], matrices['content'][1])
            cmTe = self.getContentMatrix(X_test, matrices['content'][0], matrices['content'][1])

        # Same here
        smTr, smTe = None, None
        if 'stylistic' in matrices:
            smTr = self.getStylisticMatrix(X_train, matrices['stylistic'][0], matrices['stylistic'][1])
            smTe = self.getStylisticMatrix(X_test, matrices['stylistic'][0], matrices['stylistic'][1])

        # If both we want them to be merged
        x_train, x_test = None, None
        if 'stylistic' in matrices and 'content' in matrices:
            x_train = np.concatenate((cmTr, smTr), 1)
            x_test = np.concatenate((cmTe, smTe), 1)

        # Otherwise choose the one
        if x_train is None and x_test is None:
            x_train = cmTr if cmTr is not None else smTr
            x_test = cmTe if cmTe is not None else smTe

        # Get np.array of Y_train
        y_train = np.array(Y_train)
        y_test = np.array(Y_test)

        # Return
        return x_train, y_train, x_test, y_test

    """
    getPerformance: given, y_true, y_pred and y_pred auroc, return performance metrics
    Input:  y_true: actual y labels for the corresponding matrix
            y_pred: y labels predicted from our algorithms
            y_pred_auroc: special case for auroc
    Output: dictionary of metric scores
    """
    def getPerformance(self, y_true, y_pred, y_pred_auroc):
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[0,1])
        return {
                "accuracy": metrics.accuracy_score(y_true, y_pred),
                "f1-score": metrics.f1_score(y_true, y_pred),
                "auroc": metrics.roc_auc_score(y_true, y_pred_auroc),
                "precision": metrics.precision_score(y_true, y_pred),
                "sensitivity": confusion_matrix[0][0]/ (confusion_matrix[0][0] + confusion_matrix[0][1]),
                "specificity": confusion_matrix[1][1]/ (confusion_matrix[1][0] + confusion_matrix[1][1])}

    """
    getContentMatrix(): Returns the feature matrix for content features
    Input: cols : Integer, the number of columns in the feature matrix
           n    : Integer, Describes n for word n grams
    Output: Numpy Array of dimension(number of tweets, cols)
    """
    def getContentMatrix(self, tweetSet, n, features):
        # Generate n gram ids
        ids = self.generateNgramID(n)

        # Declare np array of zeros
        fm = np.zeros((len(tweetSet), len(ids) + 8))
        # last column of word n grams is len(ids)
        temp_col = len(ids)

        # for each tweet
        # build the feature row
        for i in range(len(tweetSet)):
            t = tweetSet[i]
            tweetNgram = t.getNgram(n)

            for seq in tweetNgram:
                if seq in ids:
                    fm[i][ids[seq]] = float(tweetNgram[seq])

            # For each feature in features,
            # add the feature to the matrix if desired
            if 0 in features:
                fm[i][temp_col] = ft.getAvgEmojis()
            if 1 in features:
                fm[i][temp_col + 1] = ft.getNumURL()
            if 2 in features:
                fm[i][temp_col + 2], fm[i][temp_col + 3] = ft.getNumTags()
            if 3 in features:
                fm[i][temp_col + 4], fm[i][temp_col + 5], fm[i][temp_col + 6], dummy, dummy2 = ft.getPOSTaggedDistribution()
            if 4 in features:
                fm[i][temp_col + 7] = ft.getNumTokens()
        return fm

    """
    getStylisticMatrix(): Returns the feature matrix for stylistic features
    Input: cols : Integer, the number of columns in the feature matrix
           n    : Integer, Describes n for character n grams
    Output: Numpy Array of dimension(number of tweets, cols)
    """
    def getStylisticMatrix(self, tweetSet, n, features=[]):

        ids = self.generateNgramID(n, True) # Generate ngram ids
        fm = np.zeros((len(tweetSet), len(ids) + 34)) # make np array

        temp_col = len(ids) # Keep track of temp_col

        # For each tweet
        # Fill in the matrix accordingly
        for i in range(len(tweetSet)):
            t = tweetSet[i]
            tweetCharGram = t.getCharNgram(n)

            for seq in tweetCharGram:
                if seq in ids:
                    fm[i][ids[seq]] = tweetCharGram[seq]

            # For each feature in features
            # add the feature to the matrix if desired
            if 0 in features:
                fm[i][temp_col] = t.getAvgNumPunct()
            if 1 in features :
                fm[i][temp_col + 1] = t.getAvgWordSize()
            if 2 in features:
                fm[i][temp_col + 2] = t.getVocabSize()
            if 3 in features :
                dummy, dummy1, dummy2, fm[i][temp_col + 3], fm[i][temp_col + 4] = t.getPOSTaggedDistribution()
            if 4 in features:
                fm[i][temp_col + 5] = t.getDigitFrequency()
            if 5 in features:
                fm[i][temp_col + 6] = t.getAvgHashTagLength()
            if 6 in features :
                fm[i][temp_col + 7] = t.getAvgCapitalizations()

            # add letter frequency in this case
            if 7 in features:
                letters = t.getLetterFrequency()
                idx = 0
                for j in range(temp_col + 8, temp_col+34):
                    fm[i][j] = letters[idx]
                    idx += 1
        return fm

    """
    genGram(): Generate dictionary of ngrams or charngrams depending on
    char's boolean value
    Input:  n    : Integer describing n for word/character n grams
            char : boolean value determining whether this is a word ngram
            or char n gram
    Output: dict final_g: dictionary of ngrams or char ngrams and their occurences,
            omitting those that appeared less than 5 times
    """
    def genGram(self, n, char=False):
        g = {} # Declare dictionary
        for tweet in self.data.getAllTweets():
            gram = tweet.getNgram(n) if char==False else tweet.getCharNgram(n)
            for ngram in gram:
                if ngram in g:
                    g[ngram] += gram[ngram]
                else:
                    g[ngram] = gram[ngram]
        final_g = {}

        for gram in g:
            if g[gram] >= 5:
                final_g[gram] = g[gram]
        return final_g

    """
    genNgram(): Returns dictionary of ids for each ngram and a dictionary of ngrams
    Input:  n    : Integer, Describes n for word/character n grams
            type : Integer, 0 for word Ngram, anything other integer for character Ngrams
    Output: word_dict: Returns dictionary of ids for each ngram
            ngrams: dictionary of word/character ngrams depending on the type
    """
    def genNgram(self, n):
        return self.genGram(n)

    """
    generateCharNgram(): Returns dictionary of ids for each ngram and a dictionary of ngrams
    Input:  n    : Integer, Describes n for word/character n grams
            type : Integer, 0 for word Ngram, anything other integer for character Ngrams
    Output: word_dict: Returns dictionary of ids for each ngram
            ngrams: dictionary of word/character ngrams depending on the type
    """
    def genCharNgram(self, n):
        return self.genGram(n, True)

    """
    generateNgramID(): Returns dictionary of ids for each ngram and a dictionary of ngrams
    Input:  n    : Integer, Describes n for word/character n grams
            type : Integer, 0 for word Ngram, anything other integer for character Ngrams
    Output: ngramIds: Dictionary of ngrams with associated id
    """
    def generateNgramID(self, n, charGram=False):
        ind = 0
        ngram_dict = {}

        grams = self.genNgram(n) if charGram == False else self.genCharNgram(n)

        for seq in grams:
            if seq not in ngram_dict:
                ngram_dict[seq] = ind
                ind += 1

    
        return ngram_dict

# For testing purposes
# And running experiments
if __name__ == "__main__":
    print("Running Vectorizer.py main")

    electionTweets = "./data/2016_US_election_tweets_100k.csv"
    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    # Initialize vectorizer
    f = Vectorizer(electionTweets, electionTrolls)

    for feature in f.runKBestFeatures(100, 3):
        print(feature)

    # Run the experiment
    f.runSVM()
