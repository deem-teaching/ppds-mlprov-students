from mlprov._prov_mixin import ProvenanceEstimator
from mlprov.numpy import ndarray
from sklearn import preprocessing as orig_preprocessing


class OneHotEncoder(orig_preprocessing.OneHotEncoder, ProvenanceEstimator):
    def transform(self, X):
        raise NotImplementedError("TODO")

    def fit_transform(self, X, y=None):
        raise NotImplementedError("TODO")


class StandardScaler(orig_preprocessing.StandardScaler, ProvenanceEstimator):
    def transform(self, X, copy=None):
        raise NotImplementedError("TODO")

    def fit_transform(self, X, y=None):
        raise NotImplementedError("TODO")


def label_binarize(y, classes):
    result_wo_prov = orig_preprocessing.label_binarize(y, classes=classes)
    raise NotImplementedError("TODO")
