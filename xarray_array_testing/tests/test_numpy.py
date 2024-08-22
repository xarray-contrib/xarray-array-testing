import hypothesis.strategies as st
import numpy as np

from xarray_array_testing.creation import CreationTests


def create_numpy_array(*, shape, dtype):
    return st.builds(np.ones, shape=st.just(shape), dtype=st.just(dtype))


class TestCreationNumpy(CreationTests):
    array_type = np.ndarray
    array_module = np

    @staticmethod
    def array_strategy_fn(*, shape, dtype):
        return create_numpy_array(shape=shape, dtype=dtype)
