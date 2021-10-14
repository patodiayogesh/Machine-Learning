from scipy import io as sio
import numpy as np
import pandas as pd
import time
from sklearn.metrics import classification_report


class KNN:

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.k = 5

    def fit(self,x_train, y_train,k=5,distance='L2'):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.dist = distance

    def euclidean_distance(self,x1, x2):
        return np.linalg.norm(x1-x2)

    def manhattan_dist(self,x1,x2):
        return np.linalg.norm(x1-x2,1)

    def inf_dist(self,x1,x2):
        return np.linalg.norm(x1-x2,np.inf)

    def calculate_distance(self,x1,x2):
        if self.dist == 'L2':
            return self.euclidean_distance(x1, x2)
        if self.dist == 'L1':
            return self.manhattan_dist(x1, x2)
        if self.dist == 'L_inf':
            return self.inf_dist(x1, x2)
    def predict(self,x_test):

        predicted = []
        for x1 in x_test:
            distances = [self.calculate_distance(x1, x2) for x2 in self.x_train]
            k_distance_index = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_distance_index]
            label = max(k_labels, key=k_labels.count)
            predicted.append(label)

        return predicted


if __name__ == '__main__':

    df = pd.read_csv('compas_dataset/propublicaTrain.csv')
    df_test = pd.read_csv('compas_dataset/propublicaTest.csv')

    x_train = df.loc[:, df.columns != 'two_year_recid']
    y_train = df.loc[:, df.columns == 'two_year_recid']

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()

    x_test = df_test.loc[:, df_test.columns != 'two_year_recid'].to_numpy()
    y_test = df_test.loc[:, df_test.columns == 'two_year_recid'].to_numpy()

    knn_clf = KNN()
    knn_clf.fit(x_train,y_train,5,distance='L2')

    knn_predicted = knn_clf.predict(x_test)
    knn_predicted = np.asarray(knn_predicted)
    print(classification_report(y_test, knn_predicted))

