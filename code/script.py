#Dependencies
import os
import urllib.parse
import pandas as pd
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = train(X_train, y_train)

    prediction = test(model, X_test)

    accuracy = model.score(X_test, y_test)
    recall = metrics.recall_score(y_test, prediction)
    precision = metrics.precision_score(y_test, prediction)
    f1 = metrics.precision_score(y_test, prediction)
    # report = metrics.classification_report(y_test, prediction)
    print(accuracy)
    print(recall)
    print(precision)
    print(f1)
    # print(report)


main()