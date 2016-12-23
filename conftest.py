from __future__ import unicode_literals
import pytest
from on_the_fly import FlyVectorizer, FlySGDClassifier


@pytest.fixture(scope="function")
def vec():
    return FlyVectorizer()


@pytest.fixture(scope="function")
def avec():
    return FlyVectorizer()

@pytest.fixture()
def random_state():
    return 42

@pytest.fixture(scope="function")
def sgd(random_state):
    return FlySGDClassifier(random_state=random_state)

@pytest.fixture(scope="function")
def asgd(random_state):
    return FlySGDClassifier(random_state=random_state)

@pytest.fixture()
def data():
    return [dict(a=1, b=2, c=['x', 'y']), dict(a=1, b=2, c=['x', 'y']), dict(a=1, b=2, c=['x', 'y', 'z'])]

@pytest.fixture()
def label():
    return [0, 1, 0]

