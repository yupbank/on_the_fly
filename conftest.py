from __future__ import unicode_literals
import pytest
from on_the_fly import FlyVectorizer, FlySGD


@pytest.fixture()
def vec():
    return FlyVectorizer()


@pytest.fixture()
def avec():
    return FlyVectorizer()


@pytest.fixture()
def sgd():
    return FlySGD()


@pytest.fixture()
def asgd():
    return FlySGD()



