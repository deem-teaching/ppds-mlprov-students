from mlprov._prov_mixin import ProvenanceEstimator
from sklearn import impute as orig_impute


class SimpleImputer(orig_impute.SimpleImputer, ProvenanceEstimator):
    def transform(self, X):
        raise NotImplementedError("TODO")

    def fit_transform(self, X, y=None):
        raise NotImplementedError("TODO")
