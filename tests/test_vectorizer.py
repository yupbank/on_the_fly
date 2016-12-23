#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

import numpy as np


def test_vec_partial_fit(data, vec):
    vec.partial_fit(data)
    assert sorted(vec.feature_names_) == ['a', 'b', 'c=x', 'c=y', 'c=z']


def test_vec_partial_transform(data, vec):
    actual = vec.partial_transform(data)
    assert not actual.toarray().any()


def test_vec_partial_fit_transform(data, vec):
    d = data[0]
    actual = vec.partial_fit_transform(d)
    assert sorted(vec.feature_names_) == ['a', 'b', 'c=x', 'c=y']
    assert actual.shape == (1, 4)
    d = data[1]
    actual = vec.partial_fit_transform(d)
    assert sorted(vec.feature_names_) == ['a', 'b', 'c=x', 'c=y']
    assert actual.shape == (1, 4)
    d = data[2]
    actual = vec.partial_fit_transform(d)
    assert sorted(vec.feature_names_) == ['a', 'b', 'c=x', 'c=y', 'c=z']
    assert actual.shape == (1, 5)

def test_vecs_subset_features(data, vec):
    d = data[0]
    actual = vec.partial_fit_transform(d)
    actual = vec.subset_features(['a', 'b', 'c=x'])
    expected = range(3)
    np.testing.assert_array_equal(actual, expected)

    assert sorted(vec.feature_names_) == ['a', 'b', 'c=x', 'c=y']

def test_vec_add(data, vec, avec):
    d = data[0]
    vec.partial_fit_transform(d)
    avec.partial_fit_transform(data)
    x = vec+avec
    assert sorted(x.feature_names_) == sorted(avec.feature_names_)

    x = avec+1
    assert sorted(x.feature_names_) == sorted(avec.feature_names_)


def test_vec_divide(data, vec, avec):
    vec.partial_fit_transform(data)
    avec.partial_fit_transform(data)
    x = vec/1.0
    assert sorted(x.feature_names_) == sorted(avec.feature_names_)


def test_vecs_mean(data, vec, avec):
    d = data[0]
    vec.partial_fit_transform(d)
    avec.partial_fit_transform(data)
    new_vec = vec.mean([vec, avec])
    assert sorted(new_vec.feature_names_) == sorted(avec.feature_names_)
