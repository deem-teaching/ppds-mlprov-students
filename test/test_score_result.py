import pandas as orig_pandas
from pandas._testing import assert_frame_equal

from mlprov._prov_mixin import ScoreResult
from mlprov.numpy import ndarray


def test_score():
    score = ScoreResult(95.5, test_predictions=ndarray([1, 0, 1]), test_labels=ndarray([1, 0, 0]))
    assert isinstance(score, ScoreResult)
    assert score == 95.5
    assert_frame_equal(score.test_predictions.provenance, orig_pandas.DataFrame({"0": list(range(3))}))
    assert_frame_equal(score.test_labels.provenance, orig_pandas.DataFrame({"1": list(range(3))}))
