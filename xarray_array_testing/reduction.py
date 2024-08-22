from contextlib import nullcontext

import hypothesis.strategies as st
import xarray.testing.strategies as xrst
from hypothesis import given

from xarray_array_testing.base import DuckArrayTestMixin


class ReductionTests(DuckArrayTestMixin):
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
