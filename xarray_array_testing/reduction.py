from contextlib import nullcontext
from functools import partial

import pytest
import xarray.testing.strategies as xrst

from xarray_array_testing.base import DuckArrayTestMixin
from xarray_array_testing.decorator import delayed_given


class ReductionTests(DuckArrayTestMixin):
    @staticmethod
    def expected_errors(op, **parameters):
        return nullcontext()

    @pytest.mark.parametrize("op", ["mean", "sum", "prod", "std", "var"])
    @delayed_given(partial(xrst.variables))
    def test_variable(self, op, variable):
        with self.expected_errors(op, variable=variable):
            # compute using xr.Variable.<OP>()
            actual = getattr(variable, op)().data
            # compute using xp.<OP>(array)
            expected = getattr(self.xp, op)(variable.data)

        self.assert_equal(actual, expected)
