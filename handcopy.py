import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_score, recall_score, f1_score, roc_curve, matthews_corrcoef,\
    auc, balanced_accuracy_score
from sklearn import tree, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
import time
import pandas as pd
import matplotlib
import graphviz
matplotlib.use('TkAgg')

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
    handGesture = handGesture.astype(float)
    return handGesture


def numFolds(X, Y):
    """

    :param X: Feature set for dataset
    :param Y: Labels
    :return:
    """
    global evaluation_metric
    predictionGB, predictionKNN, predictionRF, predictionDT, predictionLR, \
    predictionADABoost = ([] for i in range( 6))
    probGB, probKNN, probRF, probDT, probLR, probADABoost = ([] for i
                                                                      in range
                                                                      (6))
    testset = list()
    n_classes = 5
    classifiers = ['NaiveBayes', 'DecisionTree', 'k-NNClassifier', 'RandomForest', 'LogisticRegression', 'ADABoost']

    folds = KFold(n_splits=10, random_state=None, shuffle=False)

    for train, test in folds.split(X):

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

    classifierGNB = GaussianNB()
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

    classifierLR = LogisticRegression(multi_class='ovr', solver='lbfgs', )
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

    adaClassifier = AdaBoostClassifier()
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


def evaluation(true):
    """

    :param true:
    :return:
    """

    global evaluation_metric
    for algo in evaluation_metric:
        print(algo)
        predict = evaluation_metric[algo]['predictions']
        evaluation_metric[algo]['confusion_matrix'] = confusion_matrix(true, predict)
        print("Confusion Matrix:")
        print(confusion_matrix(true, predict))
        evaluation_metric[algo]['accuracy'] = accuracy_score(true, predict)
        print("Accuracy:")
        print(accuracy_score(true, predict,normalize=True, sample_weight=None))

        evaluation_metric[algo]['precision'] = precision_score(true, predict)
        print("Precision:")
        print(precision_score(true, predict))
        evaluation_metric[algo]['recall'] = recall_score(true, predict)
        print("Recall:")
        print(recall_score(true, predict))
        evaluation_metric[algo]['f1Measure'] = f1_score(true, predict)
        print("F1:")
        print(f1_score(true, predict))
        evaluation_metric[algo]['mc_coefficient'] = matthews_corrcoef(true, predict)
        print("MCC:")
        print(matthews_corrcoef(true, predict))
        prob = evaluation_metric[algo]['probability']

        fpr, tpr, thresholds = roc_curve(true, prob)
        roc_auc = auc(fpr["micro"], tpr["micro"])
        print("ROC AUC:")
        print(fpr["micro"], tpr["micro"])
        plot_roc_curve(fpr, tpr, roc_auc, algo)


def scaleData(train, test):

    scaler = StandardScaler()
    scaler.fit(train)
    trainS = scaler.transform(train)

    scaler.fit(test)
    testS = scaler.transform(test)
    return trainS, testS


def plotGraph(true):
    adaClassifier = AdaBoostClassifier()
    classifierLR = LogisticRegression()
    classifierRF = RandomForestClassifier(n_estimators=100)
    classifierkNN = KNeighborsClassifier(n_neighbors=15)
    classifierDT = tree.DecisionTreeClassifier()
    classifierGNB = GaussianNB()
    # classifierSVM = svm.SVC(gamma='scale', kernel='linear')
    classifiers = [adaClassifier, classifierGNB, classifierDT, classifierLR,
                  classifierRF, classifierkNN]
    n_classes = 5
    for classifier in classifiers:
        y_score = classifier.fit(X_train, y_train).predict(X_test)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # ROC curve for a specific class here for the class 2
        roc_auc[2]


if __name__ == "__main__":

    start = time.time()
    handgesture = readData()
    handgesture = randomizeData(handgesture)
    Xhandgesture, Yhandgesture = sliceData(handgesture)

    yTrue = numFolds(Xhandgesture, Yhandgesture)
    evaluation(yTrue)
    plotGraph(yTrue)
    plt.show()
    print("Execution time: %s" % (time.time() - start))


