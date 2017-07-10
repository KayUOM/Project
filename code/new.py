import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def main():

    data = pd.read_csv('data.csv', sep=',')

    X = data['code'].astype(str)
    y = data['status'].astype(int)

    #Feature Extraction

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
    X = vectorizer.fit_transform(X)

    #Split Data Into Training and Testing

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=43)

    #Trining

    model = MultinomialNB()
    model.fit(X, y)

    #Test

    prediction = model.predict(X_test)

    #Evaluate

    accuracy = model.score(X_test, y_test)
    recall = metrics.recall_score(y_test, prediction)
    precision = metrics.precision_score(y_test, prediction)
    f1 = metrics.precision_score(y_test, prediction)
    confusion_matrix = metrics.confusion_matrix(y_test, prediction, labels=[1, 0])
    report = metrics.classification_report(y_test, prediction)

    #Print

    print("Accuracy: %.2f" % accuracy + "%")
    print("Recall: %.2f" % recall + "%")
    print("Precision: %.2f" % precision + "%")
    print("F1: %.2f" % f1 + "%")
    print(confusion_matrix)
    print(report)

main()