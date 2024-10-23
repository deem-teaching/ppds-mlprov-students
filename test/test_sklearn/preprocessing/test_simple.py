import numpy as orig_numpy
import pandas as orig_pandas
from pandas._testing import assert_frame_equal

from mlprov import pandas as pd
from mlprov.numpy import ndarray
from mlprov.sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                                          label_binarize)


def test_one_hot_encoder():
    X = ndarray([['Male', 1], ['Female', 3], ['Female', 2]])
    y = ndarray([['Female', 1], ['Male', 4]])
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    one_hot_encoder.fit(X)
    result_0 = one_hot_encoder.transform(y)
    assert orig_numpy.allclose(result_0, orig_numpy.array([[1., 0., 1., 0., 0.],
                                                           [0., 1., 0., 0., 0.]]))
    assert_frame_equal(result_0.provenance, orig_pandas.DataFrame({"1": list(range(2))}))

    result_1 = one_hot_encoder.fit_transform(X)
    assert orig_numpy.allclose(result_1, orig_numpy.array([[0., 1., 1., 0., 0.],
                                                           [1., 0., 0., 0., 1.],
                                                           [1., 0., 0., 1., 0.]]))
    assert_frame_equal(result_1.provenance, orig_pandas.DataFrame({"0": list(range(3))}))


def test_standard_scaler():
    df_train = pd.DataFrame({'A': [1, 2, 10, 5]})
    standard_scaler = StandardScaler()
    encoded_train = standard_scaler.fit_transform(df_train)
    assert isinstance(encoded_train, ndarray)
    assert_frame_equal(encoded_train.provenance, orig_pandas.DataFrame({"0": [0, 1, 2, 3]}))

    df_test = pd.DataFrame({'A': [1, 2, 3, 5]})
    encoded_test = standard_scaler.transform(df_test)
    expected = orig_numpy.array([[-1.], [-0.71428571], [-0.42857143], [0.14285714]])
    assert orig_numpy.allclose(encoded_test, expected)
    assert isinstance(encoded_test, ndarray)
    assert_frame_equal(encoded_test.provenance, orig_pandas.DataFrame({"1": [0, 1, 2, 3]}))


def test_label_binarize():
    input_label = ndarray(['yes', 'no', 'no', 'yes'])
    classes = ['yes', 'no']
    result = label_binarize(input_label, classes)
    assert orig_numpy.allclose(result, orig_numpy.array([[0], [1], [1], [0]]))
    assert_frame_equal(input_label.provenance, orig_pandas.DataFrame({"0": list(range(4))}))
    assert_frame_equal(result.provenance, orig_pandas.DataFrame({"0": list(range(4))}))
