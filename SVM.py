import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_score, recall_score, f1_score, roc_curve, matthews_corrcoef, \
    auc, balanced_accuracy_score
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import time
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

evaluation_metric = {}

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
    predictionSVM = []
    probSVM = list()
    testset = list()

    folds = KFold(n_splits=10, random_state=None, shuffle=False)

    for train, test in folds.split(X):

        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        X_train, X_test = scaleData(X_train, X_test)
        Y_predictSVM, prob = SVMClassifier(X_train, Y_train, X_test)
        print(prob)
        predictionSVM.extend(Y_predictSVM)
        probSVM.extend(prob)
        evaluation_metric['predictions'] = predictionSVM
        evaluation_metric['probability'] = probSVM
        testset.extend(Y_test)

    return testset


def featureImportance(X,Y):
    clf = ExtraTreesClassifier()
    clf.fit(X, Y)
    print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True)
    print(model.get_support())
    XAdult_new = model.transform(X)

    return XAdult_new


def sliceData(adult):
    """

    :param adult:
    :return: X is the set of features for the dataset.
    :return: Y is the set of classes in out dataset.
    """

    X = adult.values[:, 0:14]
    Y = adult.values[:, 14]
    return X, Y


def SVMClassifier(xTrain, yTrain, xTest):
    classifierSVM = svm.SVC(kernel='linear', probability=True)
    classifierSVM.fit(xTrain, yTrain)
    yPredict = classifierSVM.predict(xTest)
    yProb = classifierSVM.decision_function(xTest)

    return yPredict, yProb


def randomizeData(adult):
    adult = adult.sample(frac=1).reset_index(drop=True)
    return adult


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("SVM" + ".png")
    plt.show()


def evaluation(true):
    global evaluation_metric
    resultMetric = []

    predict = evaluation_metric['predictions']
    resultMetric.append(confusion_matrix(true, predict))
    resultMetric.append(accuracy_score(true, predict, sample_weight=None))
    resultMetric.append(balanced_accuracy_score(true, predict))
    resultMetric.append(precision_score(true, predict))
    resultMetric.append(recall_score(true, predict))
    resultMetric.append(f1_score(true, predict))
    resultMetric.append(matthews_corrcoef(true, predict, sample_weight=None))
    prob = evaluation_metric['probability']
    print(resultMetric)
    fpr, tpr, thresholds = roc_curve(true, prob)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    resultMetric.append(roc_auc)
    exit()
    plot_roc_curve(fpr, tpr, roc_auc)
    plt.show()


    return resultMetric


def calculateRates(tn,fp,fn, tp):

    tpr = tp/(tp + fn)
    fpr = fp/(fp + tn)
    return tpr, fpr


def scaleData(train, test):

    scaler = StandardScaler()
    scaler.fit(train)
    trainS = scaler.transform(train)

    scaler.fit(test)
    testS = scaler.transform(test)
    return trainS, testS


def writeFile(results):

    svmResult = pd.DataFrame(results, index=['confusion_matric', 'accuracy', 'balanced_accuracy'
                                                      'precision', 'recall',
                                                     'f1_score', 'MCC', 'roc'])
    print(svmResult)
    svmResult.to_csv('svm.csv', header=True)


if __name__ == "__main__":
    start = time.time()
    adult = readData()
    adult = randomizeData(adult)
    XAdult, YAdult = sliceData(adult)
    XAdult = featureImportance(XAdult,YAdult)
    yTrue = numFolds(XAdult, YAdult)
    result = evaluation(yTrue)
    writeFile(result)
    print("Execution time: %s" % (time.time() - start))


