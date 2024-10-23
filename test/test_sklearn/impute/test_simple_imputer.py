import numpy as orig_numpy
import pandas as orig_pandas
from pandas._testing import assert_frame_equal

import mlprov.numpy as numpy
from mlprov.numpy import ndarray
from mlprov.sklearn.impute import SimpleImputer


def test_simple_imputer():
    X = ndarray([[7, 2, 3], [4, numpy.nan, 6], [10, 5, 9]])
    y = ndarray([[numpy.nan, 2, 3], [4, numpy.nan, 6], [10, numpy.nan, 9]])
    imp_mean = SimpleImputer(missing_values=numpy.nan, strategy='mean')
    imp_mean.fit(X)
    result_0 = imp_mean.transform(y)
    assert isinstance(result_0, ndarray)
    assert orig_numpy.allclose(result_0, orig_numpy.array([[7, 2, 3], [4, 3.5, 6], [10., 3.5, 9.]]))
    assert_frame_equal(result_0.provenance, orig_pandas.DataFrame({"1": list(range(3))}))

    imp_median = SimpleImputer(missing_values=numpy.nan, strategy='median')
    result_1 = imp_median.fit_transform(X)
    assert isinstance(result_1, ndarray)
    assert orig_numpy.allclose(result_1, orig_numpy.array([[7, 2, 3], [4, 3.5, 6], [10., 5, 9.]]))
    assert_frame_equal(result_1.provenance, orig_pandas.DataFrame({"0": list(range(3))}))
