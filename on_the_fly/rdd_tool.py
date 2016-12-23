#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
import numpy as np

from base_tool import FlyVectorizer


class FlyRddBase(object):

    def __init__(self, esti):
        self.esti = esti

    def _fit_partition(self, iterator):
        raise Exception('need to implement')

    def _transform_partition(self, data):
        raise Exception('need to implement')

    def fit(self, rdd):
        self.esti = self.esti.mean(rdd.mapPartitions(self._fit_partition).collect())
        return self

    def transform(self, rdd):
        return rdd.mapPartitions(self._transform_partition).cache()

    def fit_transform(self, rdd):
        self.fit(rdd, False)
        return self.transform(rdd, False)


class RddVectorizer(FlyRddBase):
    def __init__(self, features, label, base_vec=None):
        self.features = features
        self.label = label
        self.esti = FlyVectorizer() if base_vec is None else base_vec
        self.feature_dimension_ = None
        self.label_dimension_ = None

    @property
    def feature_dimension(self):
        if not self.feature_dimension_:
            self.feature_dimension_ = self.esti.subset_features(self.features)
        return self.feature_dimension_

    @property
    def label_dimension(self):
        if not self.label_dimension_:
            self.label_dimension_ = self.esti.subset_features(self.labels)
        return self.label_dimension_

    def _fit_partition(self, data_iter):
        self.esti.partial_fit(data_iter)
        yield self.esti
    
    def _transform_partition(self, data_iter):
        data = self.esti.partial_transform(data_iter)
        yield data[:, self.feature_dimension], data[:, self.label_dimension]
    

class RddClassifier(FlyRddBase):
    def _fit_partition(self, data_iter):
        for X, y in data_iter:
            self.esti.partial_fit(X, y)
        yield self.esti

    def _transform_partition(self, data_iter):
        for X, y in data_iter:
            y_pred = self.esti.predict(X)
            yield y_pred, y

    def _score_partition(self, data_iter):
        for X, y in data_iter:
            score_value = self.esti.score(X, y)
            yield score_value, X.shape[0]
        
    def score(self, rdd):
        score_with_weight = rdd.mapPartitions(self._score_partition).collect()
        scores = map(lambda x: x[0], score_with_weight)
        weights = map(lambda x: x[1], score_with_weight)
        return np.average(scores, weights=weights)



if __name__ == "__main__":
    main()
