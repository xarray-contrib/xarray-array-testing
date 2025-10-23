import abc
from abc import ABC
from types import ModuleType

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
