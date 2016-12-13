from __future__ import unicode_literals
import pytest
from on_the_fly import FlyVectorizer, FlySGD


@pytest.fixture(scope="function")
def vec():
    return FlyVectorizer()


@pytest.fixture(scope="function")
def avec():
    return FlyVectorizer()


@pytest.fixture(scope="function")
def sgd():
    return FlySGD()


@pytest.fixture(scope="function")
def asgd():
    return FlySGD()



