import pandas as pd

from credit_scoring_observability.contracts import validate_batch
from credit_scoring_observability.make_bad_batch import make_bad_batch


def test_reference_passes_contract():
    assert validate_batch(pd.read_csv("data/reference/reference.csv"))


def test_bad_batch_is_blocked():
    assert not validate_batch(make_bad_batch())
