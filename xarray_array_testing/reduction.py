from contextlib import nullcontext
from types import ModuleType

import hypothesis.strategies as st
import numpy as np
import xarray.testing.strategies as xrst
from hypothesis import given


class ReductionTests:
    xp: ModuleType

    @staticmethod
    def array_strategy_fn(*, shape, dtype):
        raise NotImplementedError

    @staticmethod
    def assert_equal(a, b):
        np.testing.assert_allclose(a, b)

    @staticmethod
    def expected_errors(op, **parameters):
        return nullcontext()

    @given(st.data())
    def test_variable_mean(self, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        with self.expected_errors("mean", variable=variable):
            actual = variable.mean().data
            expected = self.xp.mean(variable.data)

        self.assert_equal(actual, expected)

    @given(st.data())
    def test_variable_prod(self, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        with self.expected_errors("prod", variable=variable):
            actual = variable.prod().data
            expected = self.xp.prod(variable.data)

        self.assert_equal(actual, expected)
