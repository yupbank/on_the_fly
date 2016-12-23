#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

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


def test_vecs_mean(data, vec, avec):
    d = data[0]
    vec.partial_fit_transform(d)
    avec.partial_fit_transform(data)
    new_vec = vec.mean([vec, avec])
    assert sorted(new_vec.feature_names_) == sorted(avec.feature_names_)

