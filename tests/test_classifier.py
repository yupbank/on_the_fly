#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#


import numpy as np


def test_sgd_partial_fit(data, label, vec, sgd):
    d = vec.partial_fit_transform(data)
    sgd.partial_fit(d, label, classes=[0, 1])
    assert sgd.coef_.shape[1] == d.shape[1]


def test_sgd_reorder_coef(data, label, vec, sgd):
    d = vec.partial_fit_transform(data)
    sgd.partial_fit(d, label, classes=[0, 1])
    sgd.reorder_coef([i for i in xrange(d.shape[1] + 1)])
    assert sgd.coef_.shape[1] == d.shape[1] + 1


def test_sgd_consistent(data, label, vec, sgd, asgd):
    d = vec.partial_fit_transform(data)
    sgd.partial_fit(d, label)
    asgd.partial_fit(d, label)
    np.testing.assert_array_equal(sgd.coef_, asgd.coef_)
    np.testing.assert_array_equal(sgd.intercept_, asgd.intercept_)


def test_sgd_mean(data, label, vec, sgd, asgd):
    d = vec.partial_fit_transform(data)
    sgd.partial_fit(d, label, classes=[0, 1])
    asgd.partial_fit(d, label, classes=[0, 1])

    coef = np.copy(sgd.coef_)
    intercept = np.copy(sgd.intercept_)

    new_sgd = sgd.mean([sgd, asgd])

    np.testing.assert_array_equal(new_sgd.coef_, coef)
    np.testing.assert_array_equal(new_sgd.intercept_, intercept)

def test_sgd_divide(data, label, vec, sgd):
    d = vec.partial_fit_transform(data)
    sgd.partial_fit(d, label, classes=[0, 1])
    coef = np.copy(sgd.coef_)
    intercept = np.copy(sgd.intercept_)
    x = sgd/1
    np.testing.assert_array_equal(x.coef_, coef)
    np.testing.assert_array_equal(x.intercept_, intercept)
    x = sgd/2
    np.testing.assert_array_equal(x.coef_ * 2.0, coef)
    np.testing.assert_array_equal(x.intercept_ * 2.0, intercept)

def test_sgd_add(data, label, vec, sgd):
    d = vec.partial_fit_transform(data)
    sgd.partial_fit(d, label, classes=[0, 1])
    coef = np.copy(sgd.coef_)
    intercept = np.copy(sgd.intercept_)
    x = sgd+1
    np.testing.assert_array_equal(x.coef_, coef)
    np.testing.assert_array_equal(x.intercept_, intercept)
    x = 1+sgd
    np.testing.assert_array_equal(x.coef_, coef)
    np.testing.assert_array_equal(x.intercept_, intercept)
    y = sgd+sgd
    np.testing.assert_array_equal(y.coef_, coef*2)
    np.testing.assert_array_equal(y.intercept_, intercept*2)
