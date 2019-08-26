import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_score, recall_score, f1_score, roc_curve, matthews_corrcoef,\
    auc, balanced_accuracy_score
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.multiclass import OneVsRestClassifier

import time
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier


evaluation_metric = defaultdict(list)


def readData():
    """
    The function reads csv data file created in program 1 in which the missing values
    have been handled and the continuous attributes have been discretized.

    :return: dataSet
    """
    dataSetHandGesture = pd.read_csv('allUsers.lcl.csv')
    handGesture = dataSetHandGesture.iloc[1:]
    handGesture = handGesture.replace('?', 0)
    return handGesture


def numFolds(X, Y):
    """

    :param X: Feature set for dataset
    :param Y: Labels
    :return:
    """
    global evaluation_metric
    predictionSVM = []
    probSVM = list()
    testset = list()

    folds = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

    for train, test in folds.split(X, Y):

        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]

        Y_predictSVM, prob = SVMClassifier(X_train, Y_train, X_test)
        predictionSVM.extend(Y_predictSVM)
        probSVM.extend(prob)
        evaluation_metric['predictions'] = predictionSVM
        evaluation_metric['probability'] = probSVM

        testset.extend(Y_test)

    return testset


def sliceData(handgesture):

    cols = len(handgesture.columns)
    X = handgesture.values[:, 1:cols]
    Y = handgesture.values[:, 0]
    return X, Y


def SVMClassifier(xTrain, yTrain, xTest):
    """

    :param xTrain:
    :param yTrain:
    :param xTest:
    :return:
    """

    classifierSVM = OneVsRestClassifier(svm.SVC(kernel='linear', gamma='scale', probability=True), n_jobs=-1)
    classifierSVM.fit(xTrain, yTrain)
    yPredict = classifierSVM.predict(xTest)
    probability = classifierSVM.decision_function(xTest)
    return yPredict, probability


def randomizeData(handgesture):
    """

    :param handgesture:
    :return:
    """

    handgesture = handgesture.sample(frac=1).reset_index(drop=True)
    return handgesture


def plot_roc_curve(fpr, tpr, roc_auc, algo):
    """

    :param fpr:
    :param tpr:
    :param roc_auc:
    :param algo:
    :return:
    """
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(algo+".png")
    plt.show()

def featureImportance(X, Y):
    clf = ExtraTreesClassifier()
    clf.fit(X, Y)
    model = SelectFromModel(clf, prefit=True)
    XAdult_new = model.transform(X)
    return XAdult_new

def evaluation(true):
    """

    :param true:
    :return:
    """

    global evaluation_metric
    result = []
    predict = evaluation_metric['predictions']
    result.append(confusion_matrix(true, predict))
    result.append(accuracy_score(true, predict))
    result.append(balanced_accuracy_score(true, predict))
    result.append(precision_score(true, predict))
    result.append(recall_score(true, predict))
    result.append(f1_score(true, predict))
    result.append(matthews_corrcoef(true, predict, sample_weight=None))
    print(result)
    prob = evaluation_metric['probability']
    y = pd.get_dummies(true)
    y = np.asarray(y)
    prob = np.asarray(prob)
    print(prob)
    # print(prob[:, 1])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(roc_auc[i])
    exit()


if __name__ == "__main__":

    start = time.time()
    handgesture = readData()
    handgesture = randomizeData(handgesture)
    Xhandgesture, Yhandgesture = sliceData(handgesture)
    Yhandgesture = Yhandgesture.astype('int')
    Xhandgesture = featureImportance(Xhandgesture, Yhandgesture)
    yTrue = numFolds(Xhandgesture, Yhandgesture)
    evaluation(yTrue)
    print("Execution time: %s" % (time.time() - start))


