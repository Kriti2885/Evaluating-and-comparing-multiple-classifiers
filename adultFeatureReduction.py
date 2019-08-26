"""

"""
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_score, recall_score, f1_score, roc_curve, matthews_corrcoef, \
    auc, balanced_accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import time
import pandas as pd
import matplotlib

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


def numFolds(X, Y):
    """

    :param X: Feature set for dataset
    :param Y: Labels
    :return:
    """
    global evaluation_metric
    predictionGB, predictionKNN, predictionRF, predictionDT, predictionLR, \
    predictionADABoost, predictionSVM = ([] for i in range(7))
    probGB, probKNN, probRF, probDT, probLR, probADABoost, probSVM = ([] for i
                                                                      in range
                                                                      (7))
    testset = list()

    classifiers = ['NaiveBayes', 'DecisionTree', 'k-NNClassifier',
                   'RandomForest', 'LogisticRegression', 'ADABoost']
    folds = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

    for train, test in folds.split(X, Y):

        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        X_train, X_test = scaleData(X_train, X_test)
        for classifier in classifiers:
            evaluation_metric[classifier] = {}
            if classifier == 'NaiveBayes':
                Y_predictGB, prob = gaussianNB(X_train, Y_train, X_test)
                prob = prob[:, 1]
                predictionGB.extend(Y_predictGB)
                probGB.extend(prob)
                evaluation_metric[classifier]['predictions'] = predictionGB
                evaluation_metric[classifier]['probability'] = probGB

            elif classifier == 'DecisionTree':
                Y_predictDT, prob = decisionTree(X_train, Y_train, X_test)
                predictionDT.extend(Y_predictDT)
                prob = prob[:, 1]
                probDT.extend(prob)
                evaluation_metric[classifier]['predictions'] = predictionDT
                evaluation_metric[classifier]['probability'] = probDT

            elif classifier == 'k-NNClassifier':
                Y_predictKNN, prob = kNNClassification(X_train, Y_train, X_test)
                predictionKNN.extend(Y_predictKNN)
                prob = prob[:, 1]
                probKNN.extend(prob)
                evaluation_metric[classifier]['predictions'] = predictionKNN
                evaluation_metric[classifier]['probability'] = probKNN

            elif classifier == 'RandomForest':
                Y_predictRF, prob = randomForestClassifier(X_train, Y_train, X_test)
                predictionRF.extend(Y_predictRF)
                prob = prob[:, 1]
                probRF.extend(prob)
                evaluation_metric[classifier]['predictions'] = predictionRF
                evaluation_metric[classifier]['probability'] = probRF

            elif classifier == 'LogisticRegression':
                Y_predictLR, prob = logisticRegression(X_train, Y_train, X_test)
                predictionLR.extend(Y_predictLR)
                prob = prob[:, 1]
                probLR.extend(prob)
                evaluation_metric[classifier]['predictions'] = predictionLR
                evaluation_metric[classifier]['probability'] = probLR

            elif classifier == 'ADABoost':
                Y_predictADA, prob = adaBoostClassifier(X_train, Y_train, X_test)
                predictionADABoost.extend(Y_predictADA)
                prob = prob[:, 1]
                probADABoost.extend(prob)
                evaluation_metric[classifier]['predictions'] = predictionADABoost
                evaluation_metric[classifier]['probability'] = probADABoost

        testset.extend(Y_test)

    return testset


def featureImportance(X,Y):
    clf = ExtraTreesClassifier()
    clf.fit(X, Y)
    print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    print(model.get_support())
    XAdult_new = model.transform(X)
    print(XAdult.shape)
    print(XAdult_new.shape)
    return XAdult_new


def scaleData(train, test):

    scaler = StandardScaler()
    scaler.fit(train)
    trainS = scaler.transform(train)

    scaler.fit(test)
    testS = scaler.transform(test)
    return trainS, testS


def sliceData(adult):
    """

    :param adult:
    :return: X is the set of features for the dataset.
    :return: Y is the set of classes in out dataset.
    """

    X = adult.values[:, 0:14]
    Y = adult.values[:, 14]
    return X, Y


def gaussianNB(xTrain, yTrain, xTest):
    """

    :param xTrain:
    :param yTrain:
    :param xTest:
    :return: predict is the predicted class for test data.
    """

    classifierGNB = GaussianNB()
    classifierGNB.fit(xTrain, yTrain)
    predict = classifierGNB.predict(xTest)
    probability = classifierGNB.predict_proba(xTest)

    return predict, probability


def decisionTree(xTrain, yTrain, xTest):
    classifierDT = tree.DecisionTreeClassifier()
    classifierDT.fit(xTrain, yTrain)
    yPredict = classifierDT.predict(xTest)
    probability = classifierDT.predict_proba(xTest)
    return yPredict, probability


def kNNClassification(xTrain, yTrain, xTest):
    """

    :param xTrain:
    :param yTrain:
    :param xTest:
    :return:
    """

    classifierkNN = KNeighborsClassifier(n_neighbors=15)
    classifierkNN.fit(xTrain, yTrain)
    predict = classifierkNN.predict(xTest)
    probability = classifierkNN.predict_proba(xTest)
    return predict, probability


def randomForestClassifier(xTrain, yTrain, xTest):
    classifierRF = RandomForestClassifier(n_estimators=500)
    classifierRF.fit(xTrain, yTrain)
    yPredict = classifierRF.predict(xTest)
    probability = classifierRF.predict_proba(xTest)
    return yPredict, probability


def logisticRegression(xTrain, yTrain, xTest):
    classifierLR = LogisticRegression()
    classifierLR.fit(xTrain, yTrain)
    yPredict = classifierLR.predict(xTest)
    probability = classifierLR.predict_proba(xTest)
    return yPredict, probability


def adaBoostClassifier(xTrain, yTrain, xTest):
    adaClassifier = AdaBoostClassifier()
    adaClassifier.fit(xTrain, yTrain)
    yPredict = adaClassifier.predict(xTest)
    probability = adaClassifier.predict_proba(xTest)
    return yPredict, probability


def randomizeData(adult):
    adult = adult.sample(frac=1).reset_index(drop=True)
    return adult


def plot_roc_curve(fpr, tpr, roc_auc, algo):
    plot_title = "Receiver Operating Characteristic - Adult Dataset"
    plt.plot(fpr, tpr, label=(algo + 'ROC curve (area = %0.3f)' % roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_title)
    plt.legend(loc="lower right")
    plt.savefig("adult_scale.png")


def evaluation(true):
    global evaluation_metric
    resultMetric = {}
    for algo in evaluation_metric:
        resultMetric[algo] = []
        print(algo)
        predict = evaluation_metric[algo]['predictions']
        resultMetric[algo].append(confusion_matrix(true, predict))
        resultMetric[algo].append(accuracy_score(true, predict, normalize=True, sample_weight=None))
        resultMetric[algo].append(balanced_accuracy_score(true, predict))
        resultMetric[algo].append(precision_score(true, predict))
        resultMetric[algo].append(recall_score(true, predict))
        resultMetric[algo].append(f1_score(true, predict))
        resultMetric[algo].append(matthews_corrcoef(true, predict, sample_weight=None))
        prob = evaluation_metric[algo]['probability']
        fpr, tpr, thresholds = roc_curve(true, prob)
        roc_auc = auc(fpr, tpr)
        resultMetric[algo].append(roc_auc)
        plot_roc_curve(fpr, tpr, roc_auc, algo)
    plt.show()
    return resultMetric


def writeFile(results):
    adultResultsScale = pd.DataFrame(results, index=['confusion_matric', 'accuracy',
                                                     'balanced_accuracy', 'precision', 'recall',
                                                     'f1_score', 'MCC', 'auc'])
    print(adultResultsScale)
    adultResultsScale = adultResultsScale.transpose()
    adultResultsScale.to_csv('adultScale.csv', header=True)


if __name__ == "__main__":
    start = time.time()
    adult = readData()
    adult = randomizeData(adult)
    XAdult, YAdult = sliceData(adult)
    XAdult = featureImportance(XAdult, YAdult)
    yTrue = numFolds(XAdult, YAdult)
    results = evaluation(yTrue)
    writeFile(results)
    print("Execution time: %s" % (time.time() - start))


