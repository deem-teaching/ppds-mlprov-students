import os

import pandas as orig_pandas
from pandas._testing import assert_frame_equal

import mlprov.pandas as pd
from mlprov._prov_manager import MLProvManager
from mlprov._prov_mixin import ScoreResult
from mlprov.numpy import ndarray
from mlprov.sklearn import preprocessing, tree
from mlprov.utils import get_project_root


def test_get_provenance_from_classifier():
    train_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_train.csv")
    train_data = pd.read_csv(train_file, na_values='?', index_col=0)
    test_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_test.csv")
    test_data = pd.read_csv(test_file, na_values='?', index_col=0)

    assert_frame_equal(train_data.provenance, orig_pandas.DataFrame({"0": list(range(22792))}))
    assert_frame_equal(test_data.provenance, orig_pandas.DataFrame({"1": list(range(9769))}))

    train_labels = preprocessing.label_binarize(train_data['income-per-year'], classes=['>50K', '<=50K'])
    test_labels = preprocessing.label_binarize(test_data['income-per-year'], classes=['>50K', '<=50K'])

    assert_frame_equal(train_labels.provenance, orig_pandas.DataFrame({"0": list(range(22792))}))
    assert_frame_equal(test_labels.provenance, orig_pandas.DataFrame({"1": list(range(9769))}))

    selected_train_data = train_data[["age", "hours-per-week"]]
    assert_frame_equal(selected_train_data.provenance, orig_pandas.DataFrame({"0": list(range(22792))}))

    clf = tree.DecisionTreeClassifier()
    clf.fit(selected_train_data, train_labels)

    clf_train_labels = MLProvManager().get_training_labels_for_classifier(clf)
    clf_train_data = MLProvManager().get_training_data_for_classifier(clf)
    assert isinstance(clf_train_data, pd.DataFrame)
    assert isinstance(clf_train_labels, ndarray)
    assert_frame_equal(clf_train_data.provenance, orig_pandas.DataFrame({"0": list(range(22792))}))
    assert_frame_equal(clf_train_labels.provenance, orig_pandas.DataFrame({"0": list(range(22792))}))

    selected_test_data = test_data[["age", "hours-per-week"]]
    result = clf.predict(selected_test_data)
    clf_test_data = MLProvManager().get_test_data_for_classifier(clf)
    assert isinstance(clf_test_data, pd.DataFrame)
    assert_frame_equal(clf_test_data.provenance, orig_pandas.DataFrame({"1": list(range(9769))}))

    score = clf.score(selected_test_data, test_labels)
    assert isinstance(score, ScoreResult)

    test_predictions = MLProvManager().get_test_predictions_for_score(score)
    test_labels = MLProvManager().get_test_true_labels_for_score(score)
    assert isinstance(test_predictions, ndarray)
    assert isinstance(test_labels, ndarray)
    assert_frame_equal(test_predictions.provenance, test_labels.provenance)
