import pandas as orig_pandas
from pandas._testing import assert_frame_equal

from mlprov.numpy import ndarray
from mlprov.sklearn.datasets import load_iris
from mlprov.sklearn.tree import DecisionTreeClassifier


def test_decision_tree_classifier():
    iris = load_iris()
    input_data = iris.data
    labels = iris.target
    clf = DecisionTreeClassifier()
    clf.fit(input_data, labels)

    test_data = ndarray([[4.3, 1.6, 0.3, 0.5], [5.3, 1.2, 0.5, 1.3]])
    result = clf.predict(test_data)
    assert isinstance(result, ndarray)
    assert_frame_equal(input_data.provenance, orig_pandas.DataFrame({"0": list(range(150))}))
    assert_frame_equal(labels.provenance, orig_pandas.DataFrame({"1": list(range(150))}))
    assert_frame_equal(test_data.provenance, orig_pandas.DataFrame({"2": list(range(2))}))
    assert_frame_equal(result.provenance, orig_pandas.DataFrame({"2": list(range(2))}))
