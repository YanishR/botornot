from tweet.py import Data
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
import numpy as np


electionTweetsDir = "./data/2016_US_election_tweets_100k.csv"
electionTrolls = "./data/russian-troll-tweets/IRAhandle_tweets_1.csv"


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
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1, -1])
        return confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])

    if metric == "specificity":
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1, -1])
        return confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])

    return -1

if __name__ == "__main__":
    print("hello")
    d = Data("./data/")
