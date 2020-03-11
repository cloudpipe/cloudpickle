import typing
from .mod import module_function


def package_function():
    """Function living inside a package, not a simple module"""
    return "hello from a package!"


class _SingletonClass(object):
    def __reduce__(self):
        # This reducer is only valid for the top level "some_singleton" object.
        return "some_singleton"


some_singleton = _SingletonClass()
T = typing.TypeVar('T')
