from contextlib import nullcontext

import hypothesis.strategies as st
import xarray.testing.strategies as xrst
from hypothesis import given

from xarray_array_testing.base import DuckArrayTestMixin
from xarray_array_testing.strategies import orthogonal_indexers, vectorized_indexers


def broadcast_orthogonal_indexers(indexers, sizes, *, xp):
    def _broadcasting_shape(index, total):
        return tuple(1 if i != index else -1 for i in range(total))

    def _as_array(indexer, size):
        if isinstance(indexer, slice):
            return xp.asarray(range(*indexer.indices(size)), dtype="int64")
        elif isinstance(indexer, int):
            return xp.asarray(indexer, dtype="int64")
        else:
            return indexer

    indexer_arrays = {
        dim: _as_array(indexer, sizes[dim]) for dim, indexer in indexers.items()
    }
    broadcasted = xp.broadcast_arrays(
        *(
            xp.reshape(indexer, _broadcasting_shape(index, total=len(indexers)))
            for index, indexer in enumerate(indexer_arrays.values())
        )
    )

    return dict(zip(indexer_arrays.keys(), broadcasted))


class IndexingTests(DuckArrayTestMixin):
    @property
    def orthogonal_indexer_types(self):
        return st.sampled_from(["scalars", "slices"])

    @staticmethod
    def expected_errors(op, **parameters):
        return nullcontext()

    @given(st.data())
    def test_variable_isel_orthogonal(self, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))
        idx = data.draw(orthogonal_indexers(sizes=variable.sizes, min_dims=1))

        with self.expected_errors("isel_orthogonal", variable=variable, indexers=idx):
            actual = variable.isel(idx).data

            sorted_dims = sorted(idx.keys(), key=variable.dims.index, reverse=True)
            expected = variable.data
            for dim in sorted_dims:
                indexer = idx[dim]
                axis = variable.get_axis_num(dim)
                if isinstance(indexer, slice):
                    indexer = self.xp.asarray(
                        range(*indexer.indices(variable.sizes[dim])), dtype="int64"
                    )
                expected = self.xp.take(expected, indexer, axis=axis)

        assert isinstance(
            actual, self.array_type("orthogonal_indexing")
        ), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)

    @given(st.data())
    def test_variable_isel_vectorized(self, data):
        variable = data.draw(xrst.variables(array_strategy_fn=self.array_strategy_fn))
        idx = data.draw(vectorized_indexers(sizes=variable.sizes, min_dims=1))

        with self.expected_errors("isel_vectorized", variable=variable):
            actual = variable.isel(idx).data

            raw_indexers = {dim: idx.get(dim, slice(None)) for dim in variable.dims}
            expected = variable.data[*raw_indexers.values()]

        assert isinstance(
            actual, self.array_type("vectorized_indexing")
        ), f"wrong type: {type(actual)}"
        self.assert_equal(actual, expected)
