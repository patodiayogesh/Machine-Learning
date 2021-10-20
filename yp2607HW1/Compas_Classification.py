"""
Use of MLE,KNN and NB Classifiers to predict
two year recidivism on COMPAS dataset
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

#Import models
from models.MLE import MLE
from models.KNN import KNN
from models.NB import NB

if __name__ == '__main__':

    df = pd.read_csv('compas_dataset/propublicaTrain.csv')
    df_test = pd.read_csv('compas_dataset/propublicaTest.csv')

    x_train = df.loc[:, df.columns != 'two_year_recid']
    y_train = df.loc[:, df.columns == 'two_year_recid']

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()

    x_test = df_test.loc[:, df_test.columns != 'two_year_recid'].to_numpy()
    y_test = df_test.loc[:, df_test.columns == 'two_year_recid'].to_numpy()

    # Using KNN for Classification
    knn_clf = KNN()
    knn_clf.fit(x_train, y_train, 5, distance='L2')
    knn_predicted = knn_clf.predict(x_test)
    knn_predicted = np.asarray(knn_predicted)
    print('KNN Classifier Report')
    print(classification_report(y_test, knn_predicted))

    # Using Maximum Likelihood Estimate for Classification
    mle_clf = MLE()
    mle_clf.fit(x_train, y_train)
    predictions = mle_clf.predict(x_test)
    print('MLE Classifier Report')
    print(classification_report(y_test, predictions))

    # Using Naive Bayes for Classification
    y_test = df_test.loc[:, df_test.columns == 'two_year_recid']
    df_test = df_test.loc[:, df_test.columns != 'two_year_recid']

    nb_clf = NB()
    nb_clf.fit(df, 'two_year_recid')
    predicted = nb_clf.predict(df_test)
    print('NB Classifier Report')
    print(classification_report(y_test, predicted))