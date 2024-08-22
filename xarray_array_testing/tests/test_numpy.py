import hypothesis.strategies as st
import numpy as np

from xarray_array_testing.creation import CreationTests
from xarray_array_testing.reduction import ReductionTests


def create_numpy_array(*, shape, dtype):
    return st.builds(np.ones, shape=st.just(shape), dtype=st.just(dtype))


class TestCreationNumpy(CreationTests):
    array_type = np.ndarray

    @staticmethod
    def array_strategy_fn(*, shape, dtype):
        return create_numpy_array(shape=shape, dtype=dtype)


class TestReductionNumpy(ReductionTests):
    xp = np

    @staticmethod
    def array_strategy_fn(*, shape, dtype):
        return create_numpy_array(shape=shape, dtype=dtype)
