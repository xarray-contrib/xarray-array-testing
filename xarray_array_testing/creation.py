import hypothesis.strategies as st
import xarray.testing.strategies as xrst
from hypothesis import given


class CreationTests:
    array_type: type

    @staticmethod
    def array_strategy_fn(*, shape, dtype):
        raise NotImplementedError

    @given(st.data())
    def test_create_variable(self, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        assert isinstance(variable.data, self.array_type)
