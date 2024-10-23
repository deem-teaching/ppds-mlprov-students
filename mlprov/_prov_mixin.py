from abc import ABC, abstractmethod


class PandasProvenanceMixin:
    _metadata = ["provenance"]  # Tell pandas to allow this custom attribute

    def __init__(self, *args, provenance=None, **kwargs):
        self.provenance = provenance
        super().__init__(*args, **kwargs)


# Abstract interface for any scikit-learn transformer or estimator that ensures provenance
class ProvenanceClassifier(ABC):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TODO")

    @abstractmethod
    def fit(self, X, y=None):
        """Fit method"""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict method"""
        pass

    @abstractmethod
    def score(self, X, y):
        """Score method"""
        pass


class ProvenanceEstimator(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        """Fit method that returns a result with ProvenanceMixin."""
        pass

    @abstractmethod
    def transform(self, X, y=None):
        """Transform method that returns a result with ProvenanceMixin."""
        pass

    @abstractmethod
    def fit_transform(self, X, y=None):
        """Fit and transform method that returns a result with ProvenanceMixin."""
        pass


class ScoreResult(float):
    def __new__(cls, score, test_predictions=None, test_labels=None):
        # Create a new float instance
        instance = super(ScoreResult, cls).__new__(cls, score)
        return instance

    def __init__(self, score, test_predictions=None, test_labels=None):
        raise NotImplementedError("TODO")
