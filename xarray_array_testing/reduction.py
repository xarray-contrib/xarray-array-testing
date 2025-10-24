import itertools
from contextlib import nullcontext

import hypothesis.strategies as st
import numpy as np
import pytest
import xarray.testing.strategies as xrst
from hypothesis import given, note

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

        assert isinstance(actual, self.array_type(op)), f"wrong type: {type(actual)}"
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

        assert isinstance(actual, self.array_type(op)), f"wrong type: {type(actual)}"
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

        assert isinstance(actual, self.array_type(op)), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)

    @pytest.mark.parametrize("op", ["argmax", "argmin"])
    @given(st.data())
    def test_variable_order_reduce_index(self, op, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))
        possible_dims = [..., list(variable.dims), *variable.dims] + list(
            itertools.chain.from_iterable(
                map(list, itertools.combinations(variable.dims, length))
                for length in range(1, len(variable.dims))
            )
        )
        dim = data.draw(st.sampled_from(possible_dims))

        with self.expected_errors(op, variable=variable):
            # compute using xr.Variable.<OP>()
            actual = getattr(variable, op)(dim=dim)
            if dim is ... or isinstance(dim, list):
                actual_ = {dim_: var.data for dim_, var in actual.items()}
            else:
                actual_ = actual.data

            if dim is not ... and not isinstance(dim, list):
                # compute using xp.<OP>(array)
                note(dim)
                axis = variable.get_axis_num(dim)
                indices = getattr(self.xp, op)(variable.data, axis=axis)

                expected = self.xp.asarray(indices)
            elif dim is ... or len(dim) == len(variable.dims):
                # compute using xp.<OP>(array)
                index = getattr(self.xp, op)(variable.data)

                unraveled = np.unravel_index(index, variable.shape)
                expected = {
                    k: self.xp.asarray(v) for k, v in zip(variable.dims, unraveled)
                }
            elif len(dim) == 1:
                dim_ = dim[0]
                axis = variable.get_axis_num(dim_)
                index = getattr(self.xp, op)(variable.data, axis=axis)

                expected = {dim_: self.xp.asarray(index)}
            else:
                # move the relevant dims together and flatten
                dim_name = object()
                stacked = variable.stack({dim_name: dim})

                reduce_shape = tuple(variable.sizes[d] for d in dim)
                index = getattr(self.xp, op)(stacked.data, axis=-1)

                unravelled = np.unravel_index(index, reduce_shape)

                expected = {
                    d: self.xp.asarray(idx)
                    for d, idx in zip(dim, unravelled, strict=True)
                }

            note(f"original: {variable}")
            note(f"actual: {repr(actual_)}")
            note(f"expected: {repr(expected)}")

            self.assert_dimension_indexers_equal(actual_, expected)

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

        assert isinstance(actual, self.array_type(op)), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)
