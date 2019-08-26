"""

"""
from collections import defaultdict

from sklearn import tree, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, StratifiedKFold
import time
import pandas as pd
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

evaluation_metric = defaultdict(list)


def readData():
    """
    The function reads csv data file created in program 1 in which the
    missing values have been handled and the continuous attributes have
    been discretized.

    :return: dataSet
    """

    dataSetAdult = pd.read_csv('adult.csv')
    return dataSetAdult


def sliceData(adult):
    """

    :param adult:
    :return: X is the set of features for the dataset.
    :return: Y is the set of classes in out dataset.
    """

    X = adult.values[:, 0:14]
    Y = adult.values[:, 14]
    return X, Y


def createClassifiers():
    adaClassifier = AdaBoostClassifier()
    classifierLR = LogisticRegression()
    classifierRF = RandomForestClassifier(n_estimators=100)
    classifierkNN = KNeighborsClassifier(n_neighbors=15)
    classifierDT = tree.DecisionTreeClassifier()
    classifierGNB = GaussianNB()
    classifierSVM = svm.SVC(gamma='scale', kernel='linear')
    classifier = [adaClassifier, classifierGNB, classifierDT, classifierLR,
                  classifierRF, classifierkNN, classifierSVM]
    return classifier


def createCurveMeanSquareError(X, y, classifier):

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for clf in classifier:
        print(clf)
        name = clf
        train_sizes, train_scores, validation_scores = learning_curve(
            clf, X, y, cv=cv, scoring='neg_mean_squared_error')
        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(validation_scores, axis=1)
        test_scores_std = np.std(validation_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        plt.legend(loc="best")
        plt.title(name)
        plt.show()


def createCurveAccuracy(X, y, classifier):

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for clf in classifier:
        print(clf)
        name = clf
        train_sizes, train_scores, validation_scores = learning_curve(
            clf, X, y, cv=cv, scoring='accuracy')
        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(validation_scores, axis=1)
        test_scores_std = np.std(validation_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        plt.legend(loc="best")
        plt.title(name)
        plt.show()


if __name__ == "__main__":
    start = time.time()
    adult = readData()
    XAdult, YAdult = sliceData(adult)
    classifiers = createClassifiers()
    size = len(XAdult)
    createCurveMeanSquareError(XAdult, YAdult, classifiers)
    createCurveAccuracy(XAdult, YAdult, classifiers)
    print("Execution time: %s" % (time.time() - start))


