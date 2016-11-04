from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.linear_model import SGDClassifier
from itertools import count
import scipy.sparse as sp
from collections import Mapping
from sklearn.utils.fixes import frombuffer_empty
from sklearn.utils import check_X_y
from array import array


class FlyVectorizer(DictVectorizer):
    """
    DictVectorizer that supports partial fit and transform
    """
    def add_default(self):
        if not hasattr(self, 'feature_names_') or not hasattr(self, 'vocabulary_'):
            self.vocabulary_ = dict()
            self.feature_names_ = []
        else:
            self.sort = False

    def partial_fit(self, X, y=None):
        self.add_default()

        # Process everything as sparse regardless of setting
        X = [X] if isinstance(X, Mapping) else X

        for x in X:
            for f, v in x.iteritems():
                if isinstance(v, basestring):
                    f = "%s%s%s" % (f, self.separator, v)
                if f not in self.vocabulary_:
                    self.vocabulary_[f] = len(self.feature_names_)
                    self.feature_names_.append(f)

        return self


    def partial_transform(self, X, fitting=None):
        self.add_default()

        dtype = self.dtype

        # Process everything as sparse regardless of setting
        X = [X] if isinstance(X, Mapping) else X

        indices = array("i")
        indptr = array("i", [0])
        # XXX we could change values to an array.array as well, but it
        # would require (heuristic) conversion of dtype to typecode...
        values = []

        for x in X:
            for f, v in x.iteritems():
                if isinstance(v, basestring):
                    f = "%s%s%s" % (f, self.separator, v)
                    v = 1
                if f in self.vocabulary_:
                    indices.append(self.vocabulary_[f])
                    values.append(dtype(v))
                else:
                    if fitting:
                        self.vocabulary_[f] = len(self.feature_names_)
                        self.feature_names_.append(f)
                        indices.append(self.vocabulary_[f])
                        values.append(dtype(v))

            indptr.append(len(indices))

        if len(indptr) == 1:
            raise ValueError("Sample sequence X is empty.")

        indices = frombuffer_empty(indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        shape = (len(indptr) - 1, len(self.vocabulary_))

        result_matrix = sp.csr_matrix((values, indices, indptr),
                                      shape=shape, dtype=dtype)

        # Sort everything if asked
        if fitting and self.sort:
            self.feature_names_.sort()
            map_index = np.empty(len(self.feature_names_), dtype=np.int32)
            for new_val, f in enumerate(self.feature_names_):
                map_index[new_val] = self.vocabulary_[f]
                self.vocabulary_[f] = new_val
            result_matrix = result_matrix[:, map_index]

        if self.sparse:
            result_matrix.sort_indices()
        else:
            result_matrix = result_matrix.toarray()

        return result_matrix

    def partial_fit_transform(self, X, y=None):
        return self.partial_transform(X, fitting=True)


class FlySGD(SGDClassifier):
    def fly_fit(self, X, y, classes=None, sample_weight=None):
        """
        fly classifier helps you to online or batch
        :param X:
        :param y:
        :param classes:
        :param sample_weight:
        :return:
        """
        X, y = check_X_y(X, y, "csr", copy=False, order='C', dtype=np.float64)
        if self.coef_ is not None and X.shape[1] > self.coef_.shape[1]:
            self.coef_ = np.pad(self.coef_, ((0, 0), (0, X.shape[1]-self.coef_.shape[1])), mode='constant')
        return self.partial_fit(X, y, classes, sample_weight)

    def reorder_coef(self, new_order):
        '''
        For merge cases, order before merge
        :param new_order:
        :return:
        '''
        if self.coef_ is not None:
            self.coef_ = self.coef_[:, new_order]


def test():
    f = FlyVectorizer(sparse=False)
    clf = FlySGD()
    data = [dict(a=1, b=2, c='x'), dict(a=1, b=3, c='y')]
    label = [1, 0]
    data = f.partial_fit_transform(data)
    clf.fly_fit(data, label, classes=np.array([0, 1]))
    print clf.coef_
    new_data = [dict(a=1, b=2, c='z'), dict(a=1, b=2, c='y')]
    new_label = [1, 0]
    f.sparse = True
    new_data = f.partial_fit_transform(new_data)
    print new_data.toarray()
    clf.fly_fit(new_data, new_label, classes=np.array([0, 1]))
    print clf.coef_

if __name__ == "__main__":
    test()
