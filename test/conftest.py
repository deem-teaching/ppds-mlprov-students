import pytest

from mlprov import MLProvManager


@pytest.fixture(autouse=True)
def reset_id_generator():
    prov_manager = MLProvManager()
    prov_manager.reset()