from numbers import Number
from array import array
from collections import Mapping
import types
import copy

import numpy as np
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.utils import check_X_y
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils.fixes import frombuffer_empty


class FlyBase(object):
    def add(self, other):
        raise Exception('need implementation')

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return self
        return self.add(other)
    
    def __radd__(self, other):
        return self.__add__(other)

    def div(self, other):
        raise Exception('need implementation')

    def __div__(self, other):
        return self.div(other)

    def __truediv__(self, other):
        return self.div(other)

    @staticmethod
    def mean(objs):
        return sum(objs)/len(objs)


class FlyVectorizer(DictVectorizer, FlyBase):
    """
    DictVectorizer that supports partial fit and transform.
    It supports iteratble fileds in value of dictionary.
    """

    def add_default(self):
        if not hasattr(self, 'feature_names_') or not hasattr(self, 'vocabulary_'):
            self.vocabulary_ = dict()
            self.feature_names_ = []
            self.inverse_vocabulary_ = dict()
        else:
            self.sort = False

    def partial_fit(self, X, y=None):
        self.add_default()

        X = [X] if isinstance(X, Mapping) else X
        for x in X:
            for f, v in x.iteritems():
                self.add_element(f, v)
        return self

    def add_element(self, f, v, fitting=True, transforming=False, indices=None, values=None):
        if hasattr(v, '__iter__'):
            for vv in v.__iter__():
                self.add_element(f, vv, fitting, transforming, indices, values)
        else:
            if isinstance(v, basestring):
                feature_name = "%s%s%s" % (f, self.separator, v)
                v = 1
            elif isinstance(v, (Number, types.BooleanType, types.NoneType)):
                feature_name = f
            else:
                raise Exception('Unsupported Type %s for {%s: %s}' % (type(v), f, v))
            if fitting:
                if feature_name not in self.vocabulary_:
                    self.vocabulary_[feature_name] = len(self.feature_names_)
                    self.feature_names_.append(feature_name)

            if transforming:
                if feature_name in self.vocabulary_:
                    indices.append(self.vocabulary_[feature_name])
                    values.append(self.dtype(v))

    def partial_transform(self, X, fitting=None):
        self.add_default()
        transforming = True

        # Process everything as sparse regardless of setting
        X = [X] if isinstance(X, Mapping) else X

        indices = array("i")
        indptr = array("i", [0])
        # XXX we could change values to an array.array as well, but it
        # would require (heuristic) conversion of dtype to typecode...
        values = []

        for x in X:
            for f, v in x.iteritems():
                self.add_element(f, v, fitting, transforming, indices, values)
            indptr.append(len(indices))

        if len(indptr) == 1:
            raise ValueError("Sample sequence X is empty.")

        indices = frombuffer_empty(indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        shape = (len(indptr) - 1, len(self.vocabulary_))

        result_matrix = sp.csr_matrix((values, indices, indptr),
                                      shape=shape, dtype=self.dtype)

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

    def subset_features(self, features):
        features = features or self.feature_names_
        dimensions = []
        for feature in self.feature_names_:
            if feature in features or feature.split(self.separator)[0] in features:
                dimensions.append(self.vocabulary_[feature])
        return dimensions
    
    @property
    def inverse_vocabulary(self):
        if not hasattr(self, 'inverse_vocabulary_') or not self.inverse_vocabulary_:
            self.inverse_vocabulary_ = dict((j, i) for i, j in self.vocabulary_.iteritems())
        return self.inverse_vocabulary_

    def inverse_feature_by_index(self, index):
        return self.inverse_vocabulary[index].split(self.separator)

    def add(self, other):
        new = copy.copy(self)
        new.feature_names_ = sorted(set(self.feature_names_).union(set(other.feature_names_)))
        new.vocabulary_ = dict((feature, index) for index, feature in enumerate(new.feature_names_))
        return new

    def div(self, count):
        return self
    
    def orders_from(self, lead):
        return map(lambda x: self.vocabulary_.get(x, -1), lead.feature_names_)


class FlySGDClassifier(SGDClassifier, FlyBase):

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        classifier helps you to online or batch process instances
        :param X: sample_instances, dense np.array or csr matrix
        :param y: sample_labels, dense np.array
        :param classes: [0, 1] or [-1, 1] depends on data, all the classes have to be seen for online partial fit
        :param sample_weight: instances can have different weight according to trainning
        :return: self
        """
        X, y = check_X_y(X, y, "csr", copy=False, order='C', dtype=np.float64)
        classes = np.array([0, 1]) if classes is None else classes
        if hasattr(self, 'coef_') and self.coef_ is not None and X.shape[1] > self.coef_.shape[1]:
            self.coef_ = np.pad(self.coef_, ((0, 0), (0, X.shape[1] - self.coef_.shape[1])), mode='constant')

        return super(FlySGDClassifier, self).partial_fit(X, y, classes, sample_weight)

    def reorder_coef(self, new_order):
        '''
        For merge cases, order before merge
        :param new_order: adjust the coef_ by new coordinated order
        :return: self
        '''
        if self.coef_.shape[1] < len(new_order):
            self.coef_ = np.pad(self.coef_, ((0, 0), (0, len(new_order) - self.coef_.shape[1])), mode='constant')
        self.coef_ = self.coef_[:, new_order]
        return self

    def add(self, other):
        new = clone(self)
        new.classes_ = self.classes_
        new.coef_ = np.sum(np.stack([self.coef_, other.coef_]), axis=0)
        if self.fit_intercept:
            new.intercept_ = np.sum(np.stack([self.intercept_, other.intercept_]), axis=0)
        return new

    def div(self, count):
        new = clone(self)
        new.classes_ = self.classes_
        new.coef_ = self.coef_ / count
        new.intercept_ = self.intercept_ / count
        return new



def demo():
    f = FlyVectorizer(sparse=False)
    clf = FlySGDClassifier()
    data = [dict(a=1, b=2, c='x'), dict(a=1, b=3, c='y')]
    label = [1, 0]
    data = f.partial_fit_transform(data)
    clf.partial_fit(data, label, classes=np.array([0, 1]))
    print clf.coef_
    new_data = [dict(a=1, b=2, c='z'), dict(a=1, b=2, c='y')]
    new_label = [1, 0]
    f.sparse = True
    new_data = f.partial_fit_transform(new_data)
    print new_data.toarray()
    clf.partial_fit(new_data, new_label, classes=np.array([0, 1]))
    print clf.coef_

if __name__ == "__main__":
    demo()
