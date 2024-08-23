from functools import partial

from hypothesis import given


def instantiate_given(params, **kwargs):
    def maybe_apply_kwargs(param, **kwargs):
        if not isinstance(param, partial):
            return param
        else:
            return param(**kwargs)

    given_args, given_kwargs = params
    instantiated_args = tuple(
        maybe_apply_kwargs(param, **kwargs) for param in given_args
    )
    instantiated_kwargs = {
        name: maybe_apply_kwargs(param, **kwargs)
        for name, param in given_kwargs.items()
    }

    return instantiated_args, instantiated_kwargs


def initialize_tests(cls):
    for name in dir(cls):
        if not name.startswith("test_"):
            continue

        method = getattr(cls, name)

        if not hasattr(method, "__hypothesis_given__"):
            continue
        params = method.__hypothesis_given__
        args, kwargs = instantiate_given(
            params, array_strategy_fn=cls.array_strategy_fn
        )
        decorated = given(*args, **kwargs)(method)

        setattr(cls, name, decorated)

    return cls


def delayed_given(*_given_args, **_given_kwargs):
    def wrapper(f):
        f.__hypothesis_given__ = (_given_args, _given_kwargs)

        return f

    return wrapper
