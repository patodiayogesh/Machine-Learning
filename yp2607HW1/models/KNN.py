from scipy import io as sio
import numpy as np
from sklearn.metrics import classification_report


class KNN:
    """
    K-Nearest Neighbors

    Attributes
    __________
    x_train: numpy array containing training input samples
    y_train: numpy array containing training input labels
    k: nearest neighbors
    """
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.k = 5

    def fit(self,x_train, y_train,k=5,distance='L2'):
        """
        Builds a KNN classifier from training data

        :param x_train: {numpy array} Training input samples
        :param y_train: {numpy array} Training input labels
        :param k: {int} Nearest neighbors
        :param distance: {str} Distance Metric
        :return: KNN fitted classifier
        """
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.dist = distance

        return self

    def euclidean_distance(self,x1, x2):
        """
        Function to calculate L2 metric distance

        :param x1: {numpy array} test features
        :param x2: {numpy array} train features
        :return: {numpy array} distance vector
        """
        return np.linalg.norm(x1-x2)

    def manhattan_dist(self,x1,x2):
        """
        Function to calculate L1 metric distance

        :param x1: {numpy array} test features
        :param x2: {numpy array} train features
        :return: {numpy array} distance vector
        """
        return np.linalg.norm(x1-x2,1)

    def inf_dist(self,x1,x2):
        """
        Function to calculate L inf metric distance

        :param x1: {numpy array} test features
        :param x2: {numpy array} train features
        :return: {numpy array} distance vector
        """
        return np.linalg.norm(x1-x2,np.inf)

    def calculate_distance(self,x1,x2):
        """
        Function to calculate distance between data poins

        :param x1: {numpy array} test features
        :param x2: {numpy array} train features
        :return: {numpy array} distance vector
        """

        if self.dist == 'L2':
            return self.euclidean_distance(x1, x2)
        if self.dist == 'L1':
            return self.manhattan_dist(x1, x2)
        if self.dist == 'L_inf':
            return self.inf_dist(x1, x2)

    def predict(self,x_test):
        """
        Function to predict probabilities of input dataset

        :param x_test: {numpy array} Test Dataset
        :return: {list} Predicted Labels of each datapoint
        """

        predicted = []
        for x1 in x_test:
            distances = [self.calculate_distance(x1, x2) for x2 in self.x_train]
            k_distance_index = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_distance_index]
            label = max(k_labels, key=k_labels.count)
            predicted.append(label)

        return predicted
