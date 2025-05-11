from contextlib import nullcontext

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import xarray as xr
import xarray.testing.strategies as xrst
from hypothesis import given

from xarray_array_testing.base import DuckArrayTestMixin


def scalar_indexer(size):
    return st.integers(min_value=-size, max_value=size - 1)


def integer_array_indexer(size):
    dtypes = npst.integer_dtypes()

    return npst.arrays(
        dtypes, size, elements={"min_value": -size, "max_value": size - 1}
    )


def indexers(size, indexer_types):
    indexer_strategy_fns = {
        "scalars": scalar_indexer,
        "slices": st.slices,
        "integer_arrays": integer_array_indexer,
    }

    bad_types = set(indexer_types) - indexer_strategy_fns.keys()
    if bad_types:
        raise ValueError(f"unknown indexer strategies: {sorted(bad_types)}")

    # use the order of definition to prefer simpler strategies over more complex
    # ones
    indexer_strategies = [
        strategy_fn(size)
        for name, strategy_fn in indexer_strategy_fns.items()
        if name in indexer_types
    ]
    return st.one_of(*indexer_strategies)


@st.composite
def orthogonal_indexers(draw, sizes, indexer_types):
    # TODO: make use of `flatmap` and `builds` instead of `composite`
    possible_indexers = {
        dim: indexers(size, indexer_types) for dim, size in sizes.items()
    }
    concrete_indexers = draw(xrst.unique_subset_of(possible_indexers))
    return {dim: draw(indexer) for dim, indexer in concrete_indexers.items()}


@st.composite
def vectorized_indexers(draw, sizes):
    max_size = max(sizes.values())
    shape = draw(st.integers(min_value=1, max_value=max_size))
    dtypes = npst.integer_dtypes()

    indexers = {
        dim: npst.arrays(
            dtypes, shape, elements={"min_value": -size, "max_value": size - 1}
        )
        for dim, size in sizes.items()
    }

    return {
        dim: xr.Variable("points", draw(indexer)) for dim, indexer in indexers.items()
    }


class IndexingTests(DuckArrayTestMixin):
    @property
    def orthogonal_indexer_types(self):
        return st.sampled_from(["scalars", "slices"])

    @staticmethod
    def expected_errors(op, **parameters):
        return nullcontext()

    @given(st.data())
    def test_variable_isel_orthogonal(self, data):
        indexer_types = data.draw(
            st.lists(self.orthogonal_indexer_types, min_size=1, unique=True)
        )
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))
        idx = data.draw(orthogonal_indexers(variable.sizes, indexer_types))

        with self.expected_errors(
            "isel_orthogonal", variable=variable, indexer_types=indexer_types
        ):
            actual = variable.isel(idx).data

            raw_indexers = {dim: idx.get(dim, slice(None)) for dim in variable.dims}
            expected = variable.data[*raw_indexers.values()]

        assert isinstance(actual, self.array_type), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)

    @given(st.data())
    def test_variable_isel_vectorized(self, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))
        idx = data.draw(vectorized_indexers(variable.sizes))

        with self.expected_errors("isel_vectorized", variable=variable):
            actual = variable.isel(idx).data

            raw_indexers = {dim: idx.get(dim, slice(None)) for dim in variable.dims}
            expected = variable.data[*raw_indexers.values()]

        assert isinstance(actual, self.array_type), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)
