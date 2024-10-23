"""Predicting which patients are at a higher risk of complications"""
import os
import warnings

import pandas as pd
from scikeras.wrappers import KerasClassifier
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler)

from example_pipelines.healthcare.healthcare_utils import (
    create_model, initialize_environment)
from mlprov.utils import get_project_root

warnings.filterwarnings('ignore')
initialize_environment()

COUNTIES_OF_INTEREST = ['county2', 'county3']

patients = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                    "patients.csv"), na_values='?')
histories = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                     "histories.csv"), na_values='?')

data = patients.merge(histories, on=['ssn'])
complications = data.groupby('age_group') \
    .agg(mean_complications=('complications', 'mean'))
data = data.merge(complications, on=['age_group'])
data['label'] = data['complications'] > 1.2 * data['mean_complications']
data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]
data = data[data['county'].isin(COUNTIES_OF_INTEREST)]

impute_and_one_hot_encode = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

mini_lm = SentenceTransformer('all-MiniLM-L6-v2')
embedder = FunctionTransformer(lambda item: mini_lm.encode(item.to_list()))
featurisation = ColumnTransformer(transformers=[
    ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
    ('embedder', embedder, 'last_name'),
    ('numeric', StandardScaler(), ['num_children', 'income']),
], remainder='drop')

neural_net = KerasClassifier(model=create_model, epochs=10, batch_size=1, verbose=0,
                             hidden_layer_sizes=(9, 9,), loss="binary_crossentropy")
param_grid = {'epochs': [10]}
neural_net_with_grid_search = GridSearchCV(neural_net, param_grid, cv=2)
pipeline = Pipeline([
    ('features', featurisation),
    ('learner', neural_net_with_grid_search)])

train_data, test_data = train_test_split(data)
model = pipeline.fit(train_data, train_data['label'])
test_predictions = model.predict(test_data)
print(f"Mean accuracy: {accuracy_score(test_data['label'], test_predictions)}")
