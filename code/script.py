#Dependencies
import os
import urllib.parse
import pandas as pd
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.utils import shuffle
import matplotlib.rcsetup as rcsetup
import time
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
import itertools
import numpy as np

my_path = os.path.curdir

def test(model, test):

    prediction = model.predict(test)
    return prediction

def train(X, y):

    model = MultinomialNB()
    model.fit(X, y)
    return model

def trainSVM(X, y):

    model = LinearSVC()
    model.fit(X, y)
    return model

def trainKNN(X, y):

    model = KNeighborsClassifier()
    model.fit(X, y)
    return model

def trainTree(X, y):

    model = tree.DecisionTreeClassifier()
    model.fit(X, y)
    return model

def featureExtraction(data):

    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3))
    x = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names()
    # print(feature_names)
    # weightvalue = dict(zip(vectorizer.get_feature_names(), x.data))
    # pprint(weightvalue)
    return x

def roc(actual, prediction):

    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, prediction)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.xlabel('False Positive Rate (Specificity)')
    plt.show()

def pr(actual, prediction):

    precision, recall, thresholds = precision_recall_curve(
        actual, prediction)
    area = auc(recall, precision)

    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall AUC=%0.2f' % area)
    plt.legend(loc="lower left")
    plt.savefig(my_path + './pr.png', bbox_inches = 'tight')

    plt.show()

def confusionMatrix(matrix):

    width, height = matrix.shape

    # plt.matshow(matrix)
    plt.matshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks(range(width), ["Benign", "Malicious"])
    plt.yticks(range(height), ["Benign", "Malicious"])

    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j],
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(my_path + './cm.png', bbox_inches = 'tight')
    plt.show()

def crossValidation(X, y):


    kf = KFold(n_splits=50, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print(X_train)
        model = train(X_train, y_train)
        prediction = test(model, X_test)
        accuracy = model.score(X_test, y_test)
        recall = metrics.recall_score(y_test, prediction)
        precision = metrics.precision_score(y_test, prediction)
        print(accuracy, recall, precision)


def main():
    start_time = time.time()

    data = pd.read_csv('d10k.csv', sep=',')
    # print(data.shape)
    # data = data.fillna(method='ffill')
    # print(data.isnull().any())

    # 529871
    # m = data.loc[data['status'] == 0]
    # m = m.sample(frac=1)
    # m = m.sample(n=100000)
    #
    # print(m.shape)
    #
    # n = data.loc[data['status'] == 1]
    # n = n.sample(frac=1)
    # n = n.sample(n=50000)
    #
    # print(n.shape)
    #
    # result = [m,n]
    #
    #
    #
    # data = pd.concat(result)
    #
    # data = pd.DataFrame(data=data)
    #
    # data.to_csv('d100-50k.csv', index=False)
    #
    # # print(data)

    print(data.shape)

    X = data['code'].astype(str)
    y = data['status'].astype(int)
    X = featureExtraction(X)

    # 10 fold Cross Validation Code:
    # crossValidation(X, y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=4)

    model = train(X_train, y_train)

    prediction = test(model, X_test)

    # probability = model.predict_proba(X_test)

    # print(model.classes_)

    accuracy = model.score(X_test, y_test)
    recall = metrics.recall_score(y_test, prediction)
    precision = metrics.precision_score(y_test, prediction)
    f1 = metrics.precision_score(y_test, prediction)
    confusion_matrix = metrics.confusion_matrix(y_test, prediction, labels=[1, 0])
    report = metrics.classification_report(y_test, prediction)
    #
    print("Accuracy: %.2f" % accuracy + "%")
    print("Recall: %.2f" % recall + "%")
    print("Precision: %.2f" % precision + "%")
    print("F1: %.2f" % f1 + "%")
    print (confusion_matrix)
    print(report)
    print("--- %s seconds ---" % (time.time() - start_time))

    #Confusion Matrix
    confusionMatrix(confusion_matrix)

    #Precision and Recall Curve
    pr(y_test, prediction)

    #ROC
    roc(y_test, prediction)

main()


#   print(matplotlib.get_backend())
#    print(rcsetup.all_backends)