from types import ModuleType

import hypothesis.strategies as st
import numpy as np

from xarray_array_testing.base import DuckArrayTestMixin
from xarray_array_testing.creation import CreationTests
from xarray_array_testing.indexing import IndexingTests
from xarray_array_testing.reduction import ReductionTests


def create_numpy_array(*, shape, dtype):
    return st.builds(np.ones, shape=st.just(shape), dtype=st.just(dtype))


class NumpyTestMixin(DuckArrayTestMixin):
    @property
    def xp(self) -> ModuleType:
        return np

    @property
    def array_type(self) -> type[np.ndarray]:
        return np.ndarray

    @staticmethod
    def array_strategy_fn(*, shape, dtype):
        return create_numpy_array(shape=shape, dtype=dtype)


class TestCreationNumpy(CreationTests, NumpyTestMixin):
    pass


class TestReductionNumpy(ReductionTests, NumpyTestMixin):
    pass


class TestIndexingNumpy(IndexingTests, NumpyTestMixin):
    pass
