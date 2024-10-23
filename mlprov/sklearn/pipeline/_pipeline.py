from mlprov._prov_mixin import ProvenanceEstimator
from sklearn import pipeline as orig_pipeline


class Pipeline(orig_pipeline.Pipeline, ProvenanceEstimator):
    pass
