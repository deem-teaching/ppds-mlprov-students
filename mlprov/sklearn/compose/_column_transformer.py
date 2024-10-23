from mlprov._prov_mixin import ProvenanceEstimator
from mlprov.numpy import ndarray
from sklearn import compose as orig_compose


class ColumnTransformer(orig_compose.ColumnTransformer, ProvenanceEstimator):
    def transform(self, X, y=None):
        raise NotImplementedError("TODO")

    def fit_transform(self, X, y=None):
        raise NotImplementedError("TODO")
