import hypothesis.strategies as st
import xarray.testing.strategies as xrst
from hypothesis import HealthCheck, given, settings

from xarray_array_testing.base import DuckArrayTestMixin


class CreationTests(DuckArrayTestMixin):
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    @given(st.data())
    def test_create_variable(self, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        assert isinstance(variable.data, self.array_type)
