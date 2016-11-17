import pytest
import numpy as np

@pytest.fixture()
def data():
    return [dict(a=1, b=2, c=['x', 'y']), dict(a=1, b=2, c=['x', 'y']), dict(a=1, b=2, c=['x', 'y', 'z'])]

@pytest.fixture()
def label():
    return [0, 1, 0]


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


def test_vec_averag_vecs(data, vec, avec):
    d = data[0]
    vec.partial_fit_transform(d)
    avec.partial_fit_transform(data)
    new_vec = vec.average_vecs([vec, avec])
    assert sorted(new_vec.feature_names_) == sorted(avec.feature_names_)


def test_sgd_partial_fit(data, label, vec, sgd):
    d = vec.partial_fit_transform(data)
    sgd.fly_fit(d, label, classes=[0, 1])
    assert sgd.coef_.shape[1] == d.shape[1]


def test_sgd_reorder_coef(data, label, vec, sgd):
    d = vec.partial_fit_transform(data)
    sgd.fly_fit(d, label, classes=[0, 1])
    sgd.reorder_coef([i for i in xrange(d.shape[1] + 1)])
    assert sgd.coef_.shape[1] == d.shape[1] + 1


def test_sgd_normalize(data, label, vec, sgd, asgd):
    d = vec.partial_fit_transform(data)
    sgd.fly_fit(d, label, classes=[0, 1])
    asgd.fly_fit(d, label, classes=[0, 1])
    np.testing.assert_array_equal(sgd.coef_, asgd.coef_)
    np.testing.assert_array_equal(sgd.intercept_, asgd.intercept_)

    sgd.normalize(2.0)
    np.testing.assert_array_equal(sgd.coef_ * 2.0, asgd.coef_)
    np.testing.assert_array_equal(sgd.intercept_ * 2.0, asgd.intercept_)
