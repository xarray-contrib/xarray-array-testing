from importlib.metadata import version

try:
    __version__ = version("xarray_array_testing")
except Exception:
    __version__ = "9999"
