import os

import pytest
import mlprov.numpy as np
import mlprov.pandas as pd
from mlprov.numpy import ndarray
from mlprov._prov_manager import MLProvManager
from mlprov.prov_analysis.Fairness import Fairness
from mlprov.sklearn import preprocessing
from mlprov.sklearn.compose import ColumnTransformer
from mlprov.sklearn.impute import SimpleImputer
from mlprov.sklearn.pipeline import Pipeline
from mlprov.sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlprov.sklearn.tree import DecisionTreeClassifier
from mlprov.utils import get_project_root
from mlprov.prov_analysis.DataLeakage import DataLeakage
from mlprov.sklearn.datasets import load_iris


# tests if data leakage is detected if the training dataset is used for testing
def test_data_leakage_same_train_and_test_data():
    iris = load_iris()
    input_data = iris.data
    labels = iris.target
    classifier = DecisionTreeClassifier()
    classifier.fit(input_data, labels)
    test_data = ndarray([[4.3, 1.6, 0.3, 0.5], [5.3, 1.2, 0.5, 1.3]])
    result = classifier.predict(input_data)
    score = classifier.score(input_data, result)
    original_train_data = MLProvManager().get_training_data_for_classifier(classifier)
    original_train_labels = MLProvManager().get_training_labels_for_classifier(classifier)
    original_test_data = MLProvManager().get_test_data_for_classifier(classifier)
    original_test_labels = MLProvManager().get_test_true_labels_for_score(score)
    predictions_on_test_data = MLProvManager().get_test_predictions_for_score(score)
    source_tables = MLProvManager().get_source_tables_for_classifier_and_eval(classifier, score)
    assert DataLeakage.check_data_leakage(original_train_data, original_train_labels, original_test_data, original_test_labels, source_tables) is True