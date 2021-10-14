from scipy import io as sio
import numpy as np
from sklearn.metrics import classification_report

from numpy import linalg
from math import log, pi



class MLE:
    def __init__(self):
        self.mean_vector = None
        self.cov_inverse_vector = None
        self.log_determinat_vector = None

    def get_vectors(self,train_vector, mean_vector,
                    cov_inverse_vector,
                    log_determinat_vector):

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

    def predict(self,x_test):

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

if __name__ == '__main__':
    digit_corpus = sio.loadmat('digits')
    x = digit_corpus['X']
    y = digit_corpus['Y']

    shuffle = np.random.permutation(len(x))
    x = x[shuffle]
    y = y[shuffle]

    split_length = int(0.75 * len(x))
    x_train, x_test = x[0:split_length], x[split_length:]
    y_train, y_test = y[0:split_length], y[split_length:]

    mle_clf = MLE()
    mle_clf.fit(x_train, y_train)
    predictions = mle_clf.predict(x_test)

    print(classification_report(y_test, predictions))

