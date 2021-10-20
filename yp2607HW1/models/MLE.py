import numpy as np
from numpy import linalg
from math import log, pi


class MLE:

    """
    Maximum Likelihood Estimation

    Attributes
    __________
    mean_vector: Numpy Array containing mean vector for each feature
    cov_inverse_vector: Numpy array containing covariance inverse vectors
    log_determinant_vector: Numpy array containing log determinant of cov_inverse_vector

    """

    def __init__(self):
        self.mean_vector = None
        self.cov_inverse_vector = None
        self.log_determinat_vector = None

    def get_vectors(self, train_vector, mean_vector,
                    cov_inverse_vector,
                    log_determinat_vector):
        """
        Function to calculate mean_vector, cov_inverse_vector,
        log_determinant_vector from training data

        :param train_vector: {Numpy array} Training input samples
        :param mean_vector: {Numpy array} Mean Vectors of training samples
        :param cov_inverse_vector: {Numpy array} Covariance Inverse Vectors of training samples
        :param log_determinat_vector: {Numpy array} Log Determinant Vectors of training samples
        :return: None
        """

        mean = train_vector.mean(axis=0)
        cov = ((train_vector - mean).T @ (train_vector - mean)) / \
              train_vector.shape[0]

        mean_vector.append(mean)
        cov_inverse_vector.append(np.linalg.pinv(cov))

        eigen_values = np.linalg.eig(cov)[0]
        log_det = 0
        for k in range(len(eigen_values)):
            if eigen_values[k].real > 0:
                log_det += log(eigen_values[k].real)
        log_determinat_vector.append(log_det)

    def fit(self, x_train, y_train):
        """
        Builds a MLE classifier from training data

        :param x_train: {numpy array} Training input samples
        :param y_train: {numpy array} Training input labels
        :return: MLE fitted classifier
        """

        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        mean_vector = []
        cov_inverse_vector, log_determinat_vector = [], []

        for v in y:
            index = np.where(y_train == v)[0]
            train_vector = np.take(x_train, index, 0)
            self.get_vectors(train_vector, mean_vector,
                             cov_inverse_vector,
                             log_determinat_vector)

        self.mean_vector = mean_vector
        self.cov_inverse_vector = cov_inverse_vector
        self.log_determinat_vector = log_determinat_vector

        return self

    def predict(self, x_test):
        """
        Function to predict probabilities of input dataset

        :param x_test: {numpy array} Test Dataset
        :return: {list} Predicted Labels of each datapoint
        """
        feature_dim = x_test.shape[1]
        predictions = []
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for x_vector in x_test:
            local_prediction = []
            for label in y:
                prob = -0.5 * (
                        ((x_vector - self.mean_vector[label]).T) @
                        (self.cov_inverse_vector[label]) @
                        (x_vector - self.mean_vector[label])) - \
                       0.5 * (feature_dim * log(2 * pi) + self.log_determinat_vector[label])
                local_prediction.append(prob)
            predictions.append(np.argmax(local_prediction))

        return predictions
