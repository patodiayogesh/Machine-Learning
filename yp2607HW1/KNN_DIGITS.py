from scipy import io as sio
import numpy as np
from sklearn.metrics import classification_report


class KNN:

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.k = 5

    def fit(self,x_train, y_train,k=5):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k

    def euclidean_distance(self,x1, x2):
        return np.linalg.norm(x1-x2)

    def manhattan_dist(self,x1,x2):
        return np.linalg.norm(x1-x2,1)

    def inf_dist(self,x1,x2):
        return np.linalg.norm(x1-x2,np.inf)

    def predict(self,x_test):

        predicted = []
        for x1 in x_test:
            distances = [self.euclidean_distance(x1, x2) for x2 in self.x_train]
            k_distance_index = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_distance_index]
            label = max(k_labels, key=k_labels.count)
            predicted.append(label)

        return predicted


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

    knn_clf = KNN()
    knn_clf.fit(x_train,y_train,5)

    knn_predicted = knn_clf.predict(x_test)
    knn_predicted = np.asarray(knn_predicted)
    print(classification_report(y_test, knn_predicted))
