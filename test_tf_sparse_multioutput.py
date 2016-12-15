import tensorflow as tf
from sklearn import datasets
import numpy as np
import copy
import scipy.sparse as ssp


class TFMultiOutputClassifier(object):
    def __init__(self, number_of_features, number_of_labels, batch_size=10000, learning_rate=0.001):
        self.number_of_features = number_of_features
        self.number_of_labels = number_of_labels
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def csr_matrix_to_sparse_tensor_value(self, a):
        return np.vstack(a.nonzero()).T, a.data, a.shape

    def fit(self, train_X, train_y, training_epochs=1):
        x = tf.sparse_placeholder(tf.float32, [None, self.number_of_features], name='x')
        W = tf.Variable(tf.truncated_normal([self.number_of_features, self.number_of_labels], stddev=1e-3), name='w')
        y_hat = tf.sparse_tensor_dense_matmul(x, W)
        y = tf.placeholder(tf.float32, [None, self.number_of_labels], name='y')
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_hat, y))
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        previous_error = np.inf

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in xrange(training_epochs):
                for (batch_x, batch_y) in self.batch_densify(train_X, train_y):
                    _, error, self.model = sess.run([optimizer, cost, W], feed_dict={x: batch_x, y: batch_y})
                    print error
                    if abs(previous_error - error) < 0.0001 or abs(error) < 1e-9:
                        break
                    previous_error = error

    def batch_densify(self, train_X, train_y):
        data_size = train_X.shape[0]
        batch_size = self.batch_size if self.batch_size < data_size else data_size
        indexes = list(xrange(0, data_size, batch_size))
        begin_end = zip(indexes, indexes[1:])
        for begin, end in begin_end:
            _x = train_X[begin:end, :]
            _y = train_y[begin:end, :]
            x = tf.SparseTensorValue(*self.csr_matrix_to_sparse_tensor_value(_x))
            y = _y.todense() if hasattr(train_y, 'todense') else _y
            yield x, y

    def predict(self, test_X):
        return sigmoid((np.column_stack([test_X, np.one(test_X.shape[0])]) * self.model))


def load_multilabel():
    cancer = datasets.load_breast_cancer()
    train_X = cancer.data
    y1 = cancer.target
    y2 = copy.copy(y1)
    np.random.shuffle(y2)
    y3 = copy.copy(y1)
    np.random.shuffle(y3)
    train_y = np.vstack([y1, y2, y3]).T
    return ssp.csr_matrix(train_X), train_y

def main():
    train_X, train_y = load_multilabel() 
    tt =TFMultiOutputClassifier(train_X.shape[1], train_y.shape[1], 10)
    tt.fit(train_X, train_y, 1)

if __name__ == "__main__":
    main()
