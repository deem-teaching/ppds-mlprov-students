from mlprov._prov_mixin import ProvenanceClassifier, ScoreResult
from mlprov.numpy import ndarray
from sklearn import tree as orig_tree


class DecisionTreeClassifier(orig_tree.DecisionTreeClassifier, ProvenanceClassifier):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        orig_tree.DecisionTreeClassifier.fit(self, X, y)
        raise NotImplementedError("TODO")

    def predict(self, X):
        result_wo_prov = orig_tree.DecisionTreeClassifier.predict(self, X)
        raise NotImplementedError("TODO")

    def score(self, X, y, sample_weight=None):
        result_wo_prov = orig_tree.DecisionTreeClassifier.predict(self, X)
        raise NotImplementedError("TODO")
