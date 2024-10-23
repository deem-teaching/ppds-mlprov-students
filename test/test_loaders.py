import os

import pandas as orig_pandas
from pandas._testing import assert_frame_equal

from mlprov.pandas import DataFrame, read_csv
from mlprov.utils import get_project_root


def test_read_csv():
    train_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_train.csv")
    prov_df = read_csv(train_file, na_values='?', index_col=0)
    assert len(prov_df) == 22792
    assert isinstance(prov_df, DataFrame)
    assert_frame_equal(prov_df.provenance, orig_pandas.DataFrame({"0": list(range(22792))}))
