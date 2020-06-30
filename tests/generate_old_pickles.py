"""scripts reproducing pickles used to test cloudpickle backward compat support


This file contains a few python scripts that generate pickles of canonical
objects whose pickling is supported by cloudpickle (dynamic functions, enums,
classes, modules etc). These scripts must be run with an "old" version of
cloudpickle. When testing, the generated pickle files are depickled using the
active cloudpickle branch to make sure that cloudpickle is able to depickle old
cloudpickle files.
"""
import sys

from pathlib import Path
from enum import IntEnum
from types import ModuleType
from typing import TypeVar, Generic

import cloudpickle

PYTHON_INFO = "{}_{}{}".format(
    sys.implementation.name, sys.version_info.major, sys.version_info.minor
)

PICKLE_DIRECTORY = Path(__file__).parent / "old_pickles" / PYTHON_INFO


def dump_obj(obj, filename):
    with open(str(PICKLE_DIRECTORY / filename), "wb") as f:
        cloudpickle.dump(obj, f)


def nested_function_factory():
    a = 1

    def nested_function(b):
        return a + b

    return nested_function


if __name__ == "__main__":
    PICKLE_DIRECTORY.mkdir(parents=True)

    # simple dynamic function
    def simple_func(x: int, y=1):
        return x + y

    dump_obj(simple_func, "simple_func.pkl")

    # simple dynamic class
    class SimpleClass:
        def __init__(self, attribute):
            self.attribute = attribute

    dump_obj(SimpleClass, "simple_class.pkl")

    # simple dynamic module
    dynamic_module = ModuleType("dynamic_module")
    s = """if 1:
        def f(x, y=1):
            return x + y
    """
    exec(s, vars(dynamic_module))
    assert dynamic_module.f(2, 1) == 3
    dump_obj(dynamic_module, "simple_module.pkl")

    # simple dynamic Enum
    class DynamicEnum(IntEnum):
        RED = 1
        BLUE = 2

    dump_obj(DynamicEnum, "simple_enum.pkl")

    # complex dynanic function/classes involing various typing annotations
    # supported since cloudpickle 1.4
    T = TypeVar("T")

    class MyClass(Generic[T]):
        def __init__(self, attribute: T):
            self.attribute = attribute

    dump_obj(MyClass, "class_with_type_hints.pkl")

    def add(x: MyClass[int], y: MyClass[int]):
        return MyClass(x.attribute + y.attribute)

    dump_obj([MyClass, add], "function_with_type_hints.pkl")

    # Locally defined closure
    nested_function = nested_function_factory()
    dump_obj(nested_function, "nested_function.pkl")
