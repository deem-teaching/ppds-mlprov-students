"""
Some useful utils for the project
"""
import os
import random

import numpy
from keras import Input, Sequential
from keras.src.layers import Dense


def create_model(meta, hidden_layer_sizes):
    n_features_in_ = meta["n_features_in_"]
    model = Sequential()
    model.add(Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(Dense(hidden_layer_size, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def initialize_environment():
    seed = 42
    numpy.random.seed(seed)
    random.seed(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
