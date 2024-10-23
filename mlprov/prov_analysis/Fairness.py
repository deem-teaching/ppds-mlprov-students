from typing import Callable, Dict, Union

from fairlearn.metrics import MetricFrame

import pandas as pd
from pandas._testing import assert_frame_equal


class Fairness:
    @staticmethod
    def compute_fairlearn_metric_frame(train_data, train_labels, test_data, test_labels, predictions_on_test_data,
                                       source_tables: dict, sensitive_columns: list[str],
                                       metrics: Union[Callable, Dict[str, Callable]]):
        raise NotImplementedError("TODO")
