"""Limited, best-effort test suite regarding cloudpickle backward-compat.

Cloudpickle does not officially support reading pickles files
generated with an older version of cloudpickle than the one used to read the
said pickles. However, this policy is not widely known among users that use
libraries that rely on cloudpickle such as mlflow, and is subject to confusion.

As a compromise, this script make sure cloudpickle is backward compatible for a
few canonical use cases. Cloudpicke backward-compatitibility support remains a
best-effort initiative.
"""
import pickle
import sys

import pytest

from .generate_old_pickles import PICKLE_DIRECTORY


def load_obj(filename, check_deprecation_warning='auto'):
    if check_deprecation_warning == 'auto':
        # pickles files generated with cloudpickle_fast.py on old versions of
        # coudpickle with Python < 3.8 use non-deprecated reconstructors.
        check_deprecation_warning = (sys.version_info < (3, 8))
    pickle_filepath = PICKLE_DIRECTORY / filename
    if not pickle_filepath.exists():
        pytest.skip("Could not find {}".format(str(pickle_filepath)))
    with open(str(pickle_filepath), "rb") as f:
        if check_deprecation_warning:
            msg = "A pickle file created using an old"
            with pytest.warns(UserWarning, match=msg):
                obj = pickle.load(f)
        else:
            obj = pickle.load(f)
    return obj


def test_simple_func():
    f = load_obj("simple_func.pkl")
    assert f(1) == 2
    assert f(1, 1) == 2
    assert f(2, 2) == 4


def test_simple_class():
    SimpleClass = load_obj("simple_class.pkl")
    c = SimpleClass(1)
    assert hasattr(c, "attribute")
    assert c.attribute == 1

    # test class tracking feature
    assert SimpleClass is load_obj("simple_class.pkl")


def test_dynamic_module():
    mod = load_obj("simple_module.pkl")
    assert hasattr(mod, "f")
    assert mod.f(1) == 2
    assert mod.f(1, 1) == 2
    assert mod.f(2, 2) == 4


def test_simple_enum():
    enum = load_obj("simple_enum.pkl", check_deprecation_warning=False)
    assert hasattr(enum, "RED")
    assert enum.RED == 1
    assert enum.BLUE == 2

    # test enum tracking feature
    new_enum = load_obj("simple_enum.pkl", check_deprecation_warning=False)
    assert new_enum is enum


def test_complex_class():
    SimpleClass = load_obj("class_with_type_hints.pkl")
    c = SimpleClass(1)
    assert hasattr(c, "attribute")
    assert c.attribute == 1

    # test class tracking feature
    assert SimpleClass is load_obj("class_with_type_hints.pkl")


def test_complex_function():
    MyClass, f = load_obj("function_with_type_hints.pkl")
    assert len(f.__annotations__) > 0

    a = MyClass(1)
    b = MyClass(2)

    c = f(a, b)
    assert c.attribute == 3


def test_nested_function():
    f = load_obj("nested_function.pkl")
    assert f(41) == 42
