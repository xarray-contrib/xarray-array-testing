import abc
from abc import ABC
from types import ModuleType

import numpy.testing as npt


class DuckArrayTestMixin(ABC):
    @property
    @abc.abstractmethod
    def xp() -> ModuleType:
        pass

    @property
    @abc.abstractmethod
    def array_type(self) -> type:
        pass

    @staticmethod
    @abc.abstractmethod
    def array_strategy_fn(*, shape, dtype):
        raise NotImplementedError("has to be overridden")

    @staticmethod
    def assert_equal(a, b):
        npt.assert_equal(a, b)
