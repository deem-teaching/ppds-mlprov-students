from mlprov._prov_mixin import PandasProvenanceMixin
from pandas import DataFrame as OrigDataFrame
from pandas import Series as OrigSeries


class DataFrame(OrigDataFrame, PandasProvenanceMixin):
    _metadata = ["provenance"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Generate provenance for new datasource
        raise NotImplementedError("TODO")

    def __getitem__(self, key):
        result = None
        if isinstance(key, str):
            raise NotImplementedError("TODO")
        elif isinstance(key, list) and all(isinstance(item, str) for item in key):
            raise NotImplementedError("TODO")
        elif isinstance(key, OrigSeries):
            raise NotImplementedError("TODO")
        else:
            raise NotImplementedError("TODO")
        return result


class Series(OrigSeries, PandasProvenanceMixin):
    pass
