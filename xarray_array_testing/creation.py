from functools import partial

import xarray.testing.strategies as xrst

from xarray_array_testing.base import DuckArrayTestMixin
from xarray_array_testing.decorator import delayed_given


class CreationTests(DuckArrayTestMixin):
    @delayed_given(partial(xrst.variables))
    def test_create_variable(self, variable):
        assert isinstance(variable.data, self.array_type)
