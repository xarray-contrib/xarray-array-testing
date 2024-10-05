from contextlib import nullcontext

import hypothesis.strategies as st
import pytest
import xarray.testing.strategies as xrst
from hypothesis import HealthCheck, given, note, settings

from xarray_array_testing.base import DuckArrayTestMixin


class ReductionTests(DuckArrayTestMixin):
    @staticmethod
    def expected_errors(op, **parameters):
        return nullcontext()

    # TODO understand the differing executors health check error
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    @pytest.mark.parametrize("op", ["mean", "sum", "prod", "std", "var"])
    @given(st.data())
    def test_variable_numerical_reduce(self, op, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        note(f"note: {variable}")

        with self.expected_errors(op, variable=variable):
            # compute using xr.Variable.<OP>()
            actual = getattr(variable, op)().data
            # compute using xp.<OP>(array)
            expected = getattr(self.xp, op)(variable.data)

            assert isinstance(
                actual, self.array_type
            ), f"expected {self.array_type} but got {type(actual)}"
            self.assert_equal(actual, expected)
