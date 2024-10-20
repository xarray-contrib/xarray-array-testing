from contextlib import nullcontext

import hypothesis.strategies as st
import xarray.testing.strategies as xrst
from hypothesis import given

from xarray_array_testing.base import DuckArrayTestMixin


@st.composite
def scalar_indexers(draw, sizes):
    # TODO: try to define this using builds and flatmap
    possible_indexers = {
        dim: st.integers(min_value=-size, max_value=size - 1)
        for dim, size in sizes.items()
    }
    indexers = xrst.unique_subset_of(possible_indexers)
    return {dim: draw(indexer) for dim, indexer in draw(indexers).items()}


class IndexingTests(DuckArrayTestMixin):
    @staticmethod
    def expected_errors(op, **parameters):
        return nullcontext()

    @given(st.data())
    def test_variable_scalar_isel(self, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))
        indexers = data.draw(scalar_indexers(sizes=variable.sizes))

        with self.expected_errors("scalar_isel", variable=variable):
            actual = variable.isel(indexers).data

            raw_indexers = {
                dim: indexers.get(dim, slice(None)) for dim in variable.dims
            }
            expected = variable.data[*raw_indexers.values()]

        assert isinstance(actual, self.array_type), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)
