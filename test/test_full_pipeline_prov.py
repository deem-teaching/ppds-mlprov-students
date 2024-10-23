import os

import pandas as orig_pandas
from pandas._testing import assert_frame_equal

import mlprov.numpy as np
import mlprov.pandas as pd
from mlprov._prov_mixin import ScoreResult
from mlprov.sklearn import preprocessing
from mlprov.sklearn.compose import ColumnTransformer
from mlprov.sklearn.impute import SimpleImputer
from mlprov.sklearn.pipeline import Pipeline
from mlprov.sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlprov.sklearn.tree import DecisionTreeClassifier
from mlprov.utils import get_project_root


def test_adult_complex_prov_executes():
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

    nested_income_pipeline = Pipeline([
        ('features', nested_feature_transformation),
        ('classifier', DecisionTreeClassifier())])

    nested_income_pipeline.fit(train_data, train_labels)

    print(nested_income_pipeline.score(test_data, test_labels))


def test_simplified_adult_complex_prov():
    train_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_train.csv")
    train_data = pd.read_csv(train_file, na_values='?', index_col=0)
    assert_frame_equal(train_data.provenance, orig_pandas.DataFrame({"0": list(range(22792))}))

    test_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_test.csv")
    test_data = pd.read_csv(test_file, na_values='?', index_col=0)
    assert_frame_equal(test_data.provenance, orig_pandas.DataFrame({"1": list(range(9769))}))

    train_labels = preprocessing.label_binarize(train_data['income-per-year'], classes=['>50K', '<=50K'])
    assert_frame_equal(train_labels.provenance, orig_pandas.DataFrame({"0": list(range(22792))}))
    test_labels = preprocessing.label_binarize(test_data['income-per-year'], classes=['>50K', '<=50K'])
    assert_frame_equal(test_labels.provenance, orig_pandas.DataFrame({"1": list(range(9769))}))

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
    assert_frame_equal(transformed_train_data.provenance, orig_pandas.DataFrame({"0": list(range(22792))}))
    classifier.fit(transformed_train_data, train_labels)

    transformed_test_data = nested_feature_transformation.fit_transform(test_data)
    assert_frame_equal(transformed_test_data.provenance, orig_pandas.DataFrame({"1": list(range(9769))}))
    score = classifier.score(transformed_test_data, test_labels)
    isinstance(score, ScoreResult)
    assert len(score.test_predictions.provenance) == 9769
    assert len(score.test_labels.provenance) == 9769
