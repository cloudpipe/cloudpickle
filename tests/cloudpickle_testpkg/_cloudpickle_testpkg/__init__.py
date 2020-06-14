import typing
from . import mod  # noqa


def package_function():
    """Function living inside a package, not a simple module"""
    return "hello from a package!"


class _SingletonClass(object):
    def __reduce__(self):
        # This reducer is only valid for the top level "some_singleton" object.
        return "some_singleton"


def relative_imports_factory():
    """Factory creating dynamically-defined functions using relative imports

    Relative import of functions living both inside modules and packages are
    tested.
    """
    def f():
        # module_function belongs to _cloudpickle_testpkg.mod, which is a
        # module
        from .mod import module_function
        return module_function()

    def g():
        # package_function belongs to _cloudpickle_testpkg, which is a package
        from . import package_function
        return package_function()

    return f, g


some_singleton = _SingletonClass()
T = typing.TypeVar('T')
