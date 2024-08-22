from contextlib import nullcontext

import hypothesis.strategies as st
import xarray.testing.strategies as xrst
from hypothesis import HealthCheck, given, note, settings

from xarray_array_testing.base import DuckArrayTestMixin


class ReductionTests(DuckArrayTestMixin):
    @staticmethod
    def expected_errors(op, **parameters):
        return nullcontext()

    # TODO understand the differing executors health check error
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    @given(st.data())
    def test_variable_mean(self, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        note(f"note: {variable}")

        with self.expected_errors("mean", variable=variable):
            actual = variable.mean().data
            expected = self.xp.mean(variable.data)

            assert isinstance(actual, self.array_type), type(actual)
            self.assert_equal(actual, expected)

    @settings(suppress_health_check=[HealthCheck.differing_executors])
    @given(st.data())
    def test_variable_prod(self, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        with self.expected_errors("prod", variable=variable):
            actual = variable.prod().data
            expected = self.xp.prod(variable.data)

            assert isinstance(actual, self.array_type), type(actual)
            self.assert_equal(actual, expected)
