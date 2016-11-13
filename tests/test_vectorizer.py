import pytest

@pytest.fixture()
def data():
    return [dict(a=1, b=2, c=['x', 'y']), dict(a=1, b=2, c=['x', 'y']), dict(a=1, b=2, c=['x', 'y', 'z'])]

def test_partial_fit(data, vec):
    vec.partial_fit(data)
    assert sorted(vec.feature_names_) == ['a', 'b', 'c=x', 'c=y', 'c=z']

def test_partial_transform(data, vec):
    actual = vec.partial_transform(data)
    assert not actual.toarray().any()

def test_partial_fit_transform(data, vec):
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

