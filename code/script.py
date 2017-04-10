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


def test(model, test):

    prediction = model.predict(test)
    return prediction

def train(X, y):

    model = MultinomialNB()
    model.fit(X, y)
    return model

def featureExtraction(data):

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
    x = vectorizer.fit_transform(data)
    return x

def main():

    data = pd.read_csv('data.csv', sep=',')

    X = data['code'].astype(str)
    y = data['status'].astype(int)
    X = featureExtraction(X)

    # kf = KFold(n_splits=10, random_state=None, shuffle=False)
    # for train_index, test_index in kf.split(X):
    #
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     model = train(X_train, y_train)
    #     prediction = test(model, X_test)
    #     accuracy = model.score(X_test, y_test)
    #     recall = metrics.recall_score(y_test, prediction)
    #     precision = metrics.precision_score(y_test, prediction)
    #     print(accuracy, recall, precision)
    #



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = train(X_train, y_train)

    prediction = test(model, X_test)

    probability = model.predict_proba(X_test)

    # print(model.classes_)

    accuracy = model.score(X_test, y_test)
    recall = metrics.recall_score(y_test, prediction)
    precision = metrics.precision_score(y_test, prediction)
    f1 = metrics.precision_score(y_test, prediction)

    confusion_matrix = metrics.confusion_matrix(y_test, prediction, labels=[1, 0])
    report = metrics.classification_report(y_test, prediction)

        print("Accuracy: %.2f" % accuracy + "%")
        print(("Recall: %.2f" % recall + "%"))
        print(("Precision: %.2f" % precision + "%"))
        print(("F1: %.2f" % f1 + "%"))
        print (confusion_matrix)

    #
    # plt.matshow(confusion_matrix)
    # plt.colorbar()
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.show()



    # print(matplotlib.get_backend())
    # print(rcsetup.all_backends)

    # print(report)


main()