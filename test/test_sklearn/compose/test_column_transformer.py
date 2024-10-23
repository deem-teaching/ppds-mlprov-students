import pandas as orig_pandas
from pandas._testing import assert_frame_equal

import mlprov.numpy as numpy
from mlprov.numpy import ndarray
from mlprov.sklearn.compose import ColumnTransformer
from mlprov.sklearn.impute import SimpleImputer
from mlprov.sklearn.preprocessing import OneHotEncoder


def test_column_transformer():
    X = ndarray([[0., numpy.nan, 'Male'],
                 [1., 1., 'Female']])
    ct = ColumnTransformer(transformers=[
        ('imputing', SimpleImputer(missing_values=numpy.nan, strategy='mean'), slice(0, 1)),
        ('onehot encoding', OneHotEncoder(handle_unknown='ignore'), slice(2))
    ])
    result = ct.fit_transform(X)
    assert isinstance(result, ndarray)
    assert_frame_equal(result.provenance, orig_pandas.DataFrame({"0": list(range(2))}))
