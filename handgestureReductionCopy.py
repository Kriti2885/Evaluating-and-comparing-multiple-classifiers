import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_score, recall_score, f1_score, roc_curve, matthews_corrcoef, \
    auc
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp
import time
import pandas as pd
import numpy as np
import matplotlib
from yellowbrick.classifier import ROCAUC

matplotlib.use('TkAgg')

evaluation_metric = defaultdict(list)
resultMetric = defaultdict(list)

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
    predictionGB, predictionKNN, predictionRF, predictionDT, predictionLR, \
    predictionADABoost = ([] for i in range(6))
    probGB, probKNN, probRF, probDT, probLR, probADABoost = ([] for i
                                                             in range
                                                             (6))
    testset = list()

    classifiers = ['NaiveBayes']
    #'NaiveBayes', 'DecisionTree', 'k-NNClassifier', 'RandomForest', 'LogisticRegression', 'ADABoost']

    folds = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

    y = label_binarize(Y, classes=[0, 1, 2, 3, 4])
    n_classes = y.shape[1]

    for train, test in folds.split(X, Y):

        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]

        X_train, X_test = scaleData(X_train, X_test)

        for classifier in classifiers:
            evaluation_metric[classifier] = {}
            if classifier == 'NaiveBayes':
                Y_predictGB, prob = gaussianNB(X_train, Y_train, X_test)
                predictionGB.extend(Y_predictGB)
                probGB.extend(prob)
                evaluation_metric[classifier]['predictions'] = predictionGB
                evaluation_metric[classifier]['probability'] = probGB

            elif classifier == 'DecisionTree':
                Y_predictDT, prob = decisionTree(X_train, Y_train, X_test)
                predictionDT.extend(Y_predictDT)
                probDT.extend(prob)
                evaluation_metric[classifier]['predictions'] = predictionDT
                evaluation_metric[classifier]['probability'] = probDT

            elif classifier == 'k-NNClassifier':
                Y_predictKNN, prob = kNNClassification(X_train, Y_train, X_test)
                predictionKNN.extend(Y_predictKNN)
                probKNN.extend(prob)
                evaluation_metric[classifier]['predictions'] = predictionKNN
                evaluation_metric[classifier]['probability'] = probKNN

            elif classifier == 'RandomForest':
                Y_predictRF, prob = randomForestClassifier(X_train, Y_train, X_test)
                predictionRF.extend(Y_predictRF)
                probRF.extend(prob)
                evaluation_metric[classifier]['predictions'] = predictionRF
                evaluation_metric[classifier]['probability'] = probRF

            elif classifier == 'LogisticRegression':
                Y_predictLR, prob = logisticRegression(X_train, Y_train, X_test)
                predictionLR.extend(Y_predictLR)
                probLR.extend(prob)
                evaluation_metric[classifier]['predictions'] = predictionLR
                evaluation_metric[classifier]['probability'] = probLR

            elif classifier == 'ADABoost':
                Y_predictADA, prob = adaBoostClassifier(X_train, Y_train, X_test)
                predictionADABoost.extend(Y_predictADA)
                probADABoost.extend(prob)
                evaluation_metric[classifier]['predictions'] = predictionADABoost
                evaluation_metric[classifier]['probability'] = probADABoost

        testset.extend(Y_test)

    return testset


def sliceData(handgesture):
    cols = len(handgesture.columns)
    X = handgesture.values[:, 1:cols]
    Y = handgesture.values[:, 0]
    return X, Y


def gaussianNB(xTrain, yTrain, xTest):
    """

    :param xTrain:
    :param yTrain:
    :param xTest:
    :return: predict is the predicted class for test data.
    """

    classifierGNB = OneVsRestClassifier(GaussianNB())
    classifierGNB.fit(xTrain, yTrain)
    predict = classifierGNB.predict(xTest)
    probability = classifierGNB.predict_proba(xTest)

    return predict, probability


def decisionTree(xTrain, yTrain, xTest):
    """

    :param xTrain:
    :param yTrain:
    :param xTest:
    :return:
    """
    classifierDT = tree.DecisionTreeClassifier()
    classifierDT.fit(xTrain, yTrain)
    yPredict = classifierDT.predict(xTest)
    probability = classifierDT.predict_proba(xTest)
    # print(classifierDT.feature_importances_)
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
    """

    :param xTrain:
    :param yTrain:
    :param xTest:
    :return:
    """
    classifierRF = RandomForestClassifier(n_estimators=100)
    classifierRF.fit(xTrain, yTrain)
    yPredict = classifierRF.predict(xTest)
    probability = classifierRF.predict_proba(xTest)

    return yPredict, probability


def logisticRegression(xTrain, yTrain, xTest):
    """

    :param xTrain:
    :param yTrain:
    :param xTest:
    :return:
    """

    classifierLR = OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver='lbfgs'))
    classifierLR.fit(xTrain, yTrain)
    yPredict = classifierLR.predict(xTest)
    probability = classifierLR.predict_proba(xTest)

    return yPredict, probability


def adaBoostClassifier(xTrain, yTrain, xTest):
    """

    :param xTrain:
    :param yTrain:
    :param xTest:
    :return:
    """

    adaClassifier = OneVsRestClassifier(AdaBoostClassifier())
    adaClassifier.fit(xTrain, yTrain)
    yPredict = adaClassifier.predict(xTest)
    probability = adaClassifier.predict_proba(xTest)

    return yPredict, probability


def randomizeData(handgesture):
    """

    :param handgesture:
    :return:
    """

    handgesture = handgesture.sample(frac=1).reset_index(drop=True)
    return handgesture


def plot(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    oz = ROCAUC(GaussianNB())
    oz.fit(X_train, y_train)
    oz.score(X_test, y_test)
    oz.poof()


def evaluation(true):
    """

    :param true:
    :return:
    """

    global evaluation_metric
    global resultMetric
    for algo in evaluation_metric:
        resultMetric[algo] = []
        print(algo)
        predict = evaluation_metric[algo]['predictions']
        resultMetric[algo].append(confusion_matrix(true, predict))
        cm = confusion_matrix(true, predict)
        cm = np.asarray(cm)
        print(cm)
        resultMetric[algo].append(accuracy_score(true, predict, normalize=True, sample_weight=None))
        resultMetric[algo].append(precision_score(true, predict, average='weighted'))
        resultMetric[algo].append(recall_score(true, predict, average='weighted'))
        resultMetric[algo].append(f1_score(true, predict, average='weighted'))
        resultMetric[algo].append(matthews_corrcoef(true, predict, sample_weight=None))
        print(resultMetric[algo])
        prob = evaluation_metric[algo]['probability']
        y = pd.get_dummies(true)
        y = np.asarray(y)
        prob = np.asarray(prob)
        print(prob)
        #print(prob[:, 1])

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(5):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            print(roc_auc[i])
            plot_roc_curve(fpr[i], tpr[i], roc_auc[i], algo)
            exit()
    return resultMetric


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


def scaleData(train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    trainS = scaler.transform(train)

    scaler.fit(test)
    testS = scaler.transform(test)
    return trainS, testS


def cleanData(data):
    return data


def featureImportance(X, Y):
    clf = ExtraTreesClassifier()
    clf.fit(X, Y)
    model = SelectFromModel(clf, prefit=True)
    XAdult_new = model.transform(X)
    return XAdult_new


def writeFile():
    global resultMetric
    handgestureResultsScale = pd.DataFrame(resultMetric, index=['confusion_matric', 'accuracy',
                                                           'precision', 'recall',
                                                           'f1_score', 'MCC'])
    print(handgestureResultsScale)
    handgestureResultsScale = handgestureResultsScale.transpose()
    handgestureResultsScale.to_csv('handgestureresults.csv', header=True)


if __name__ == "__main__":
    start = time.time()
    handgesture = readData()
    handgesture = cleanData(handgesture)
    handgesture = randomizeData(handgesture)
    Xhandgesture, Yhandgesture = sliceData(handgesture)
    Yhandgesture = Yhandgesture.astype('int')
    Xhandgesture = featureImportance(Xhandgesture, Yhandgesture)
    yTrue = numFolds(Xhandgesture, Yhandgesture)
    result = evaluation(yTrue)
    plot(Xhandgesture, Yhandgesture)
    exit()
    writeFile(result)
    print("Execution time: %s" % (time.time() - start))


