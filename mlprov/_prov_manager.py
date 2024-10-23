from mlprov._prov_mixin import ProvenanceClassifier, ScoreResult


class MLProvManager:
    _instance = None
    _next_table_id = 0
    _all_input_tables = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLProvManager, cls).__new__(cls)
        return cls._instance

    def get_training_data_for_classifier(self, classifier: ProvenanceClassifier):
        raise NotImplementedError("TODO")

    def get_training_labels_for_classifier(self, classifier: ProvenanceClassifier):
        raise NotImplementedError("TODO")

    def get_test_data_for_classifier(self, classifier: ProvenanceClassifier):
        raise NotImplementedError("TODO")

    def get_test_predictions_for_score(self, score: ScoreResult):
        raise NotImplementedError("TODO")

    def get_test_true_labels_for_score(self, score: ScoreResult):
        raise NotImplementedError("TODO")

    def get_source_tables_for_classifier_and_eval(
            self, classifier: ProvenanceClassifier, score: ScoreResult
    ):
        raise NotImplementedError("TODO")

    def get_next_table_id(self):
        raise NotImplementedError("TODO")

    def register_input_table(self, table: any):
        raise NotImplementedError("TODO")

    def reset(self):
        self._next_table_id = 0
        self._all_input_tables = []
