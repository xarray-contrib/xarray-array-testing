from contextlib import nullcontext

import hypothesis.strategies as st
import xarray.testing.strategies as xrst
from hypothesis import given

from xarray_array_testing.base import DuckArrayTestMixin


def scalar_indexer(size):
    return st.integers(min_value=-size, max_value=size - 1)


@st.composite
def indexers(draw, sizes, indexer_strategy_fn):
    possible_indexers = {dim: indexer_strategy_fn(size) for dim, size in sizes.items()}
    indexers = draw(xrst.unique_subset_of(possible_indexers))
    return {dim: draw(indexer) for dim, indexer in indexers.items()}


class IndexingTests(DuckArrayTestMixin):
    @staticmethod
    def expected_errors(op, **parameters):
        return nullcontext()

    @given(st.data())
    def test_variable_isel_scalars(self, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))
        idx = data.draw(indexers(variable.sizes, scalar_indexer))

        with self.expected_errors("isel_scalars", variable=variable):
            actual = variable.isel(idx).data

            raw_indexers = {dim: idx.get(dim, slice(None)) for dim in variable.dims}
            expected = variable.data[*raw_indexers.values()]

        assert isinstance(actual, self.array_type), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)

    @given(st.data())
    def test_variable_isel_slices(self, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))
        idx = data.draw(indexers(variable.sizes, st.slices))

        with self.expected_errors("isel_slices", variable=variable):
            actual = variable.isel(idx).data

            raw_indexers = {dim: idx.get(dim, slice(None)) for dim in variable.dims}
            expected = variable.data[*raw_indexers.values()]

        assert isinstance(actual, self.array_type), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)
