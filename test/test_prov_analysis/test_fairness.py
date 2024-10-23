import os

import pytest
from fairlearn.metrics import false_negative_rate

import mlprov.numpy as np
import mlprov.pandas as pd
from mlprov._prov_manager import MLProvManager
from mlprov.prov_analysis.Fairness import Fairness
from mlprov.sklearn import preprocessing
from mlprov.sklearn.compose import ColumnTransformer
from mlprov.sklearn.impute import SimpleImputer
from mlprov.sklearn.pipeline import Pipeline
from mlprov.sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlprov.sklearn.tree import DecisionTreeClassifier
from mlprov.utils import get_project_root


def test_simplified_adult_complex_fairness():
    train_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_train.csv")
    train_data = pd.read_csv(train_file, na_values='?', index_col=0)
    test_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_test.csv")
    test_data = pd.read_csv(test_file, na_values='?', index_col=0)

    train_labels = preprocessing.label_binarize(train_data['income-per-year'], classes=['>50K', '<=50K'])
    test_labels = preprocessing.label_binarize(test_data['income-per-year'], classes=['>50K', '<=50K'])

    nested_categorical_feature_transformation = Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    nested_feature_transformation = ColumnTransformer(transformers=[
        ('categorical', nested_categorical_feature_transformation, ['education', 'workclass']),
        ('numeric', StandardScaler(), ['age', 'hours-per-week'])
    ])

    classifier = DecisionTreeClassifier()

    transformed_train_data = nested_feature_transformation.fit_transform(train_data, train_labels)
    classifier.fit(transformed_train_data, train_labels)

    transformed_test_data = nested_feature_transformation.fit_transform(test_data)
    score = classifier.score(transformed_test_data, test_labels)

    original_train_data = MLProvManager().get_training_data_for_classifier(classifier)
    original_train_labels = MLProvManager().get_training_labels_for_classifier(classifier)
    original_test_data = MLProvManager().get_test_data_for_classifier(classifier)
    original_test_labels = MLProvManager().get_test_true_labels_for_score(score)
    predictions_on_test_data = MLProvManager().get_test_predictions_for_score(score)
    source_tables = MLProvManager().get_source_tables_for_classifier_and_eval(classifier, score)

    fnr = Fairness.compute_fairlearn_metric_frame(train_data=original_train_data,
                                                  train_labels=original_train_labels,
                                                  test_data=original_test_data,
                                                  test_labels=original_test_labels,
                                                  predictions_on_test_data=predictions_on_test_data,
                                                  source_tables=source_tables, sensitive_columns=["sex"],
                                                  metrics=false_negative_rate)
    fnr_by_group = fnr.by_group.to_dict()

    expected_result = {'Female': 0.12, 'Male': 0.15}
    assert len(fnr_by_group) == 2
    assert fnr_by_group['Female'] == pytest.approx(expected_result['Female'], abs=1.)
    assert fnr_by_group['Male'] == pytest.approx(expected_result['Male'], abs=1.)
