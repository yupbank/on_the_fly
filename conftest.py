from __future__ import unicode_literals
import pytest
from on_the_fly import FlyVectorizer


@pytest.fixture()
def vec():
    return FlyVectorizer()


