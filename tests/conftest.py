import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df():
    return pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list("ABCD"))
