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
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

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
    # print(len(feature_names))
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
    plt.savefig(my_path + './roc.png', bbox_inches = 'tight')
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

def kFold(X, y):

    kf = KFold(n_splits=100, random_state=None, shuffle=False)
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

def crossValidation(X,y):

    clf = MultinomialNB()
    scoring = ['accuracy', 'precision', 'recall', 'f1']


    for s in scoring:

        scores = cross_val_score(clf, X, y, cv=10, scoring=s)
        print(s + " : ")
        print(np.round(scores,2))
        print(s + ": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """

    train_sizes = [100, 500, 1000, 2500, 5000, 8000]
    plt.figure()
    plt.title(title)
    print(ylim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()


    print(train_sizes)
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
    return plt

def test_training_prediction_time(X_train, y_train, X_test):

    training_time = time.time()
    model = train(X_train, y_train)
    print("--- %s Training Time (seconds) ---" % (time.time() - training_time))

    prediction_time = time.time()
    prediction = test(model, X_test)
    print("--- %s Prediction Time (seconds) ---" % (time.time() - prediction_time))

    objects = ('Training Time', 'Prediction Time')
    y_pos = np.arange(len(objects))
    performance = [4, 0.8]

    plt.bar(y_pos, performance, align='center', alpha=1)
    plt.xticks(y_pos, objects)
    plt.ylabel('Time Taken (ms)')
    plt.title('Training and Prediction Times')

    plt.show()


def main():
    start_time = time.time()

    data = pd.read_csv('d10k.csv', sep=',')
    #
    # data = data.sample(frac=1)
    # data = data.sample(n=10000)
    # m = data.loc[data['status'] == 1]
    # nm = data.loc[data['status'] == 0]
    #
    # print(m.shape)
    # print(nm.shape)


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

    print(y_test.shape, prediction.shape)



    # print(prediction[:,0])
    # plt.scatter(y_test, prediction[:,0])
    # plt.xticks([0,1])
    # plt.xlabel("Actual")
    # plt.ylabel("Predictions")
    # plt.show()


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


    # crossValidation(X,y)


    # kFold(X, y)


    # Training and Prediction Time
    # test_training_prediction_time(X_train, y_train, X_test)


    #Confusion Matrix
    # confusionMatrix(confusion_matrix)

    #Precision and Recall Curve
    # pr(y_test, prediction)

    #ROC
    # roc(y_test, prediction)

    #LEARNING CURVE
    # title = "Learning Curve"
    # # Cross validation with 100 iterations to get smoother mean test and train
    # # score curves, each time with 20% data randomly selected as a validation set.
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    #
    # estimator = MultinomialNB()
    # plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    # plt.show()

    # print(prediction)
    # plt.scatter(y_test, prediction)
    # plt.xlabel("TrueValues")
    # plt.ylabel("Predictions")
    # plt.show()

main()


