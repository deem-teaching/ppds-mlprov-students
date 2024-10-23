import os

import pandas as orig_pandas
from pandas._testing import assert_frame_equal

import mlprov.pandas as prov_pandas
from mlprov.pandas import DataFrame, Series, read_csv
from mlprov.utils import get_project_root


def test_dataframe_projection():
    train_file = os.path.join(
        str(get_project_root()), "example_pipelines", "adult_complex", "adult_train.csv"
    )
    prov_df = read_csv(train_file, na_values="?", index_col=0)
    prov_series = prov_df["age"]
    assert isinstance(prov_series, Series)
    assert_frame_equal(prov_series.provenance, orig_pandas.DataFrame({"0": list(range(22792))}))


def test_dataframe_selection():
    prov_df = prov_pandas.DataFrame({'age': [20, 50, 13, 52]})
    prov_df_selected = prov_df[prov_df["age"] > 30]
    assert isinstance(prov_df_selected, DataFrame)
    assert len(prov_df_selected) < len(prov_df)
    assert_frame_equal(prov_df_selected.provenance, orig_pandas.DataFrame({"0": [1, 3]}))


def test_dataframe_init():
    prov_df = prov_pandas.DataFrame({'age': [20, 50, 13, 52]})
    assert len(prov_df) == 4
    assert isinstance(prov_df, DataFrame)
    assert_frame_equal(prov_df.provenance, orig_pandas.DataFrame({"0": list(range(4))}))
