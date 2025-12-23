from collections.abc import Hashable
from itertools import compress

import hypothesis.extras.numpy as npst
import hypothesis.strategies as st
import numpy as np
import xarray as xr
from xr.testing.strategies import unique_subset_of


def _basic_indexers(size):
    return st.one_of(
        st.integers(min_value=-size, max_value=size - 1),
        st.slices(size),
    )


def _outer_array_indexers(size, max_size):
    return npst.arrays(
        dtype=np.int64,
        shape=st.integers(min_value=1, max_value=min(size, max_size)),
        elements=st.integers(min_value=-size, max_value=size - 1),
    )


# vendored from `xarray`, should be included in `xarray>=2026.01.0`
@st.composite
def basic_indexers(
    draw,
    /,
    *,
    sizes: dict[Hashable, int],
    min_dims: int = 1,
    max_dims: int | None = None,
) -> dict[Hashable, int | slice]:
    """Generate basic indexers using ``hypothesis.extra.numpy.basic_indices``.

    Parameters
    ----------
    draw : callable
    sizes : dict[Hashable, int]
        Dictionary mapping dimension names to their sizes.
    min_dims : int, optional
        Minimum number of dimensions to index.
    max_dims : int or None, optional
        Maximum number of dimensions to index.

    Returns
    -------
    sizes : mapping of hashable to int or slice
        Indexers as a dict with keys randomly selected from ``sizes.keys()``.

    See Also
    --------
    hypothesis.strategies.slices
    """
    selected_dims = draw(unique_subset_of(sizes, min_size=min_dims, max_size=max_dims))

    # Generate one basic index (int or slice) per selected dimension
    idxr = {
        dim: draw(
            st.one_of(
                st.integers(min_value=-size, max_value=size - 1),
                st.slices(size),
            )
        )
        for dim, size in selected_dims.items()
    }
    return idxr


@st.composite
def outer_array_indexers(
    draw,
    /,
    *,
    sizes: dict[Hashable, int],
    min_dims: int = 0,
    max_dims: int | None = None,
    max_size: int = 10,
) -> dict[Hashable, np.ndarray]:
    """Generate outer array indexers (vectorized/orthogonal indexing).

    Parameters
    ----------
    draw : callable
        The Hypothesis draw function (automatically provided by @st.composite).
    sizes : dict[Hashable, int]
        Dictionary mapping dimension names to their sizes.
    min_dims : int, optional
        Minimum number of dimensions to index
    max_dims : int or None, optional
        Maximum number of dimensions to index

    Returns
    -------
    sizes : mapping of hashable to np.ndarray
        Indexers as a dict with keys randomly selected from ``sizes.keys()``.
        Values are 1D numpy arrays of integer indices for each dimension.

    See Also
    --------
    hypothesis.extra.numpy.arrays
    """
    selected_dims = draw(unique_subset_of(sizes, min_size=min_dims, max_size=max_dims))
    idxr = {
        dim: draw(
            npst.arrays(
                dtype=np.int64,
                shape=st.integers(min_value=1, max_value=min(size, max_size)),
                elements=st.integers(min_value=-size, max_value=size - 1),
            )
        )
        for dim, size in selected_dims.items()
    }
    return idxr


@st.composite
def orthogonal_indexers(
    draw,
    /,
    *,
    sizes: dict[Hashable, int],
    min_dims: int = 2,
    max_dims: int | None = None,
    max_size: int = 10,
) -> dict[Hashable, int | slice | np.ndarray]:
    selected_dims = draw(unique_subset_of(sizes, min_size=min_dims, max_size=max_dims))

    return {
        dim: draw(
            st.one_of(
                _basic_indexers(size),
                _outer_array_indexers(size, max_size),
            )
        )
        for dim, size in selected_dims.items()
    }


@st.composite
def vectorized_indexers(
    draw,
    /,
    *,
    sizes: dict[Hashable, int],
    min_dims: int = 2,
    max_dims: int | None = None,
    min_ndim: int = 1,
    max_ndim: int = 3,
    min_size: int = 1,
    max_size: int = 5,
) -> dict[Hashable, xr.DataArray]:
    """Generate vectorized (fancy) indexers where all arrays are broadcastable.

    In vectorized indexing, all array indexers must have compatible shapes
    that can be broadcast together, and the result shape is determined by
    broadcasting the indexer arrays.

    Parameters
    ----------
    draw : callable
        The Hypothesis draw function (automatically provided by @st.composite).
    sizes : dict[Hashable, int]
        Dictionary mapping dimension names to their sizes.
    min_dims : int, optional
        Minimum number of dimensions to index. Default is 2, so that we always have a "trajectory".
        Use ``outer_array_indexers`` for the ``min_dims==1`` case.
    max_dims : int or None, optional
        Maximum number of dimensions to index.
    min_ndim : int, optional
        Minimum number of dimensions for the result arrays.
    max_ndim : int, optional
        Maximum number of dimensions for the result arrays.
    min_size : int, optional
        Minimum size for each dimension in the result arrays.
    max_size : int, optional
        Maximum size for each dimension in the result arrays.

    Returns
    -------
    sizes : mapping of hashable to DataArray or Variable
        Indexers as a dict with keys randomly selected from sizes.keys().
        Values are DataArrays of integer indices that are all broadcastable
        to a common shape.

    See Also
    --------
    hypothesis.extra.numpy.arrays
    """
    selected_dims = draw(unique_subset_of(sizes, min_size=min_dims, max_size=max_dims))

    # Generate a common broadcast shape for all arrays
    # Use min_ndim to max_ndim dimensions for the result shape
    result_shape = draw(
        st.lists(
            st.integers(min_value=min_size, max_value=max_size),
            min_size=min_ndim,
            max_size=max_ndim,
        )
    )
    result_ndim = len(result_shape)

    # Create dimension names for the vectorized result
    vec_dims = tuple(f"vec_{i}" for i in range(result_ndim))

    # Generate array indexers for each selected dimension
    # All arrays must be broadcastable to the same result_shape
    idxr = {}
    for dim, size in selected_dims.items():
        array_shape = draw(
            npst.broadcastable_shapes(
                shape=tuple(result_shape),
                min_dims=min_ndim,
                max_dims=result_ndim,
            )
        )

        # For xarray broadcasting, drop dimensions where size differs from result_shape
        # (numpy broadcasts size-1, but xarray requires matching sizes or missing dims)
        # Right-align array_shape with result_shape for comparison
        aligned_dims = vec_dims[-len(array_shape) :] if array_shape else ()
        aligned_result = result_shape[-len(array_shape) :] if array_shape else []
        keep_mask = [s == r for s, r in zip(array_shape, aligned_result, strict=True)]
        filtered_shape = tuple(compress(array_shape, keep_mask))
        filtered_dims = tuple(compress(aligned_dims, keep_mask))

        # Generate array of valid indices for this dimension
        indices = draw(
            npst.arrays(
                dtype=np.int64,
                shape=filtered_shape,
                elements=st.integers(min_value=-size, max_value=size - 1),
            )
        )
        idxr[dim] = xr.Variable(data=indices, dims=filtered_dims)
    return idxr
