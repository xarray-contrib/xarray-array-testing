from contextlib import nullcontext

import hypothesis.strategies as st
import numpy as np
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
    def test_variable_numerical_reduce(self, op, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        with self.expected_errors(op, variable=variable):
            # compute using xr.Variable.<OP>()
            actual = getattr(variable, op)().data
            # compute using xp.<OP>(array)
            expected = getattr(self.xp, op)(variable.data)

        assert isinstance(actual, self.array_type), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)

    @pytest.mark.parametrize("op", ["all", "any"])
    @given(st.data())
    def test_variable_boolean_reduce(self, op, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        with self.expected_errors(op, variable=variable):
            # compute using xr.Variable.<OP>()
            actual = getattr(variable, op)().data
            # compute using xp.<OP>(array)
            expected = getattr(self.xp, op)(variable.data)

        assert isinstance(actual, self.array_type), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)

    @pytest.mark.parametrize("op", ["max", "min"])
    @given(st.data())
    def test_variable_order_reduce(self, op, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        with self.expected_errors(op, variable=variable):
            # compute using xr.Variable.<OP>()
            actual = getattr(variable, op)().data
            # compute using xp.<OP>(array)
            expected = getattr(self.xp, op)(variable.data)

        assert isinstance(actual, self.array_type), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)

    @pytest.mark.parametrize("op", ["argmax", "argmin"])
    @given(st.data())
    def test_variable_order_reduce_index(self, op, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        with self.expected_errors(op, variable=variable):
            # compute using xr.Variable.<OP>()
            actual = {k: v.item() for k, v in getattr(variable, op)(dim=...).items()}

            # compute using xp.<OP>(array)
            index = getattr(self.xp, op)(variable.data)
            unraveled = np.unravel_index(index, variable.shape)
            expected = dict(zip(variable.dims, unraveled))

        self.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "op",
        [
            "cumsum",
            pytest.param(
                "cumprod",
                marks=pytest.mark.skip(reason="not yet included in the array api"),
            ),
        ],
    )
    @given(st.data())
    def test_variable_cumulative_reduce(self, op, data):
        array_api_names = {"cumsum": "cumulative_sum", "cumprod": "cumulative_prod"}
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))

        with self.expected_errors(op, variable=variable):
            # compute using xr.Variable.<OP>()
            actual = getattr(variable, op)().data
            # compute using xp.<OP>(array)
            # Variable implements n-d cumulative ops by iterating over dims
            expected = variable.data
            for axis in range(variable.ndim):
                expected = getattr(self.xp, array_api_names[op])(expected, axis=axis)

        assert isinstance(actual, self.array_type), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)
