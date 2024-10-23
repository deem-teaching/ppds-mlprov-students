import pandas as orig_pandas
from pandas._testing import assert_frame_equal

from mlprov.numpy import ndarray


def test_ndarray_init():
    array = ndarray(list(range(3)))
    assert isinstance(array, ndarray)
    assert_frame_equal(array.provenance, orig_pandas.DataFrame({"0": list(range(3))}))

    array = ndarray([[0, 1], [0, 1]])
    assert isinstance(array, ndarray)
    assert_frame_equal(array.provenance, orig_pandas.DataFrame({"1": [0, 1]}))
