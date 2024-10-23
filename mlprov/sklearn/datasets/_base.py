from sklearn.datasets import load_iris as orig_load_iris


def load_iris():
    result_wo_prov = orig_load_iris()
    raise NotImplementedError("TODO")
