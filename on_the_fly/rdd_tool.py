#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
from itertools import tee
import numpy as np

from base_tool import FlyVectorizer


class FlyRddBase(object):

    def __init__(self, estimator):
        self.estimator = estimator

    def _fit_partition(self, iterator):
        raise Exception('need to implement')

    def _transform_partition(self, data):
        raise Exception('need to implement')

    def fit(self, rdd):
        self.estimator = self.estimator.mean(rdd.mapPartitions(self._fit_partition).collect())
        return self

    def transform(self, rdd):
        return rdd.mapPartitions(self._transform_partition).cache()

    def fit_transform(self, rdd):
        self.fit(rdd, False)
        return self.transform(rdd, False)


class RddVectorizer(FlyRddBase):
    def __init__(self, features, label, row_index_names=None, base_vec=None):
        self.features = features
        self.label = label
        self.row_index_names = row_index_names
        self.estimator = FlyVectorizer() if base_vec is None else base_vec
        self.feature_dimension_ = None
        self.label_dimension_ = None

    @property
    def feature_dimension(self):
        if not self.feature_dimension_:
            self.feature_dimension_ = self.estimator.subset_features(self.features)
        return self.feature_dimension_

    @property
    def label_dimension(self):
        if not self.label_dimension_:
            self.label_dimension_ = self.estimator.subset_features(self.label)
        return self.label_dimension_

    def _fit_partition(self, data_iter):
        self.estimator.partial_fit(data_iter)
        yield self.estimator
    
    def _transform_partition(self, data_iter):
        if self.row_index_names is not None:
            names, data = tee(data_iter)
            row_index = map(lambda r: r.get(self.row_index_names), names)
        else:
            row_index = None
        data = self.estimator.partial_transform(data)
        yield row_index, data[:, self.feature_dimension], data[:, self.label_dimension]
    
    def get_feature_by_index(self, index):
        return self.estimator.get_feature_by_index(index)

    def dumps(self):
        return self.estimator.feature_names_, self.features, self.label, self.row_index_names

    @classmethod
    def loads(cls, (feature_names, features, features_to_rank, row_index_names)):
        estimator = FlyVectorizer()
        estimator.feature_names_ = feature_names
        estimator.vocabulary_ = dict((j, i) for i, j in enumerate(feature_names))
        obj = cls(estimator, features, features_to_rank, row_index_names)
        return obj


class RddClassifier(FlyRddBase):
    def _fit_partition(self, data_iter):
        for row_index, X, y in data_iter:
            self.estimator.partial_fit(X, y)
        yield self.estimator

    def _transform_partition(self, data_iter):
        for row_index, X, y in data_iter:
            y_pred = self.estimator.predict(X)
            yield y_pred, y

    def _score_partition(self, data_iter):
        for row_index, X, y in data_iter:
            score_value = self.estimator.score(X, y)
            yield score_value, X.shape[0]
        
    def score(self, rdd):
        score_with_weight = rdd.mapPartitions(self._score_partition).collect()
        return np.average(zip(*score_with_weight))


