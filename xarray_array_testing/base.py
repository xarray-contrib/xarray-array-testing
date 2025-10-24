import abc
from abc import ABC
from types import ModuleType

import numpy as np
import numpy.testing as npt
from xarray.namedarray._typing import duckarray


class DuckArrayTestMixin(ABC):
    @property
    @abc.abstractmethod
    def xp() -> ModuleType:
        pass

    @staticmethod
    def array_type(op: str) -> type[duckarray]:
        pass

    @staticmethod
    @abc.abstractmethod
    def array_strategy_fn(*, shape, dtype):
        raise NotImplementedError("has to be overridden")

    @staticmethod
    def assert_equal(a, b):
        npt.assert_equal(a, b)

    @staticmethod
    def assert_dimension_indexers_equal(a, b):
        assert type(a) is type(b), f"types don't match: {type(a)} vs {type(b)}"

        if isinstance(a, dict):
            assert a.keys() == b.keys(), f"Different dimensions: {list(a)} vs {list(b)}"

            assert all(np.equal(a[k], b[k]) for k in a), "Differing indexers"
        else:
            npt.assert_equal(a, b)
