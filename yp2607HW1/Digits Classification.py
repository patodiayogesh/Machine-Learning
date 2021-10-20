"""
Use of MLE and KNN Classifiers to predict
handwritten digits
Trained on digits.mat dataset
"""

from scipy import io as sio
import numpy as np
from sklearn.metrics import classification_report

#Import models
from models.MLE import MLE
from models.KNN import KNN

if __name__ == '__main__':

    digit_corpus = sio.loadmat('digits')
    x = digit_corpus['X']
    y = digit_corpus['Y']

    #Split Train and Test Data
    shuffle = np.random.permutation(len(x))
    x = x[shuffle]
    y = y[shuffle]

    split_length = int(0.75 * len(x))
    x_train, x_test = x[0:split_length], x[split_length:]
    y_train, y_test = y[0:split_length], y[split_length:]

    #Using Maximum Likelihood Estimate for Classification
    mle_clf = MLE()
    mle_clf.fit(x_train, y_train)
    predictions = mle_clf.predict(x_test)
    print('MLE Classifier Report')
    print(classification_report(y_test, predictions))

    #Using KNN for Classifiction
    knn_clf = KNN()
    knn_clf.fit(x_train, y_train, 5)
    knn_predicted = knn_clf.predict(x_test)
    knn_predicted = np.asarray(knn_predicted)
    print('KNN Classifier Report')
    print(classification_report(y_test, knn_predicted))
