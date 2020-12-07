from dataparser import Data
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
import numpy as np
from Vectorizer import Vectorizer


"""
Calculates the performance metric as evaluated on the true labels
y_true versus the predicted labels y_pred.
Input:
    y_true: (n,) array containing known labels
    y_pred: (n,) array containing predicted scores
    metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
Returns:
    the performance as an numpy float
"""
def performance(y_true, y_pred, metric="accuracy"):
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)

    if metric == "f1-score":
        return metrics.f1_score(y_true, y_pred)

    if metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)

    if metric == "precision":
        return metrics.precision_score(y_true, y_pred)

    if metric == "sensitivity":
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
        return confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])

    if metric == "specificity":
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
        return confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
    return -1

def results(X_train, Y_train, X_test, Y_test, r = 0.1, hyper_param = 0.1, kernel_type = 0, degree = 1):
    svc = SVC(C = hyper_param, kernel = 'linear', degree = degree, class_weight = 'balanced')\
            if kernel_type == 0\
            else SVC(C = hyper_param, kernel = 'poly', degree = degree, class_weight = 'balanced', coef0 = r)

    metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]

    for metric in metrics:
        svc.fit(X_train, Y_train)
        Y_predicted = svc.predict(X_test) if metric != "auroc" else svc.decision_function(X_test)
        score = performance(Y_test, Y_predicted, metric)
        print(metric + " : " + str(score))

if __name__ == "__main__":
    print("Running botornot")

    electionTweets = "./data/2016_US_election_tweets_100k.csv"
    electionTrolls = "./data/IRAhandle_tweets_1.csv"

    v = Vectorizer(Data(electionTweets, electionTrolls))

    X_train, Y_train, X_test, Y_test = v.getSplitData(3)

    results(X_train, Y_train, X_test, Y_test)
    # results(X_train, Y_train, X_test, Y_test, r = 0.1, hyper_param = 0.1, kernel_type = 1, degree = 2) #Poly kernel
