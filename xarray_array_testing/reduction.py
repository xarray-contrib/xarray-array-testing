from contextlib import nullcontext

import hypothesis.strategies as st
import pytest
import xarray.testing.strategies as xrst
from hypothesis import given

from xarray_array_testing.base import DuckArrayTestMixin


class ReductionTests(DuckArrayTestMixin):
    @staticmethod
    def expected_errors(op, **parameters):
        return nullcontext()

    @pytest.mark.parametrize("op", ["mean", "sum", "prod", "std", "var"])
    @given(st.data())
    def test_variable_mean(self, op, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        with self.expected_errors(op, variable=variable):
            # compute using xr.Variable.<OP>()
            actual = getattr(variable, op)().data
            # compute using xp.<OP>(array)
            expected = getattr(self.xp, op)(variable.data)

        self.assert_equal(actual, expected)
