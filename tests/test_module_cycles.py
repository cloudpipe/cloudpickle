"""Regression tests for modules serialized by value with cyclic state."""

import builtins
import importlib
import pickle
import sys
import types

import pytest

from cloudpickle import dumps, register_pickle_by_value, unregister_pickle_by_value

from .testutils import subprocess_pickle_echo


@pytest.mark.parametrize("protocol", [2, 4, 5])
@pytest.mark.parametrize("registered", [False, True])
def test_module_self_reference(protocol, registered, monkeypatch):
    module = types.ModuleType("cloudpickle_cycle_test")
    module.self = module
    module.nested = {"module": module}
    exec("def absolute(x): return abs(x)", module.__dict__)
    original_builtins = module.__builtins__
    if registered:
        monkeypatch.setitem(sys.modules, module.__name__, module)
        register_pickle_by_value(module)
    try:
        for restored, alias in (
            pickle.loads(dumps([module, module], protocol=protocol)),
            subprocess_pickle_echo([module, module], protocol=protocol),
        ):
            assert restored is alias
            assert restored is not module
            assert restored.self is restored
            assert restored.nested["module"] is restored
            assert restored.__builtins__["abs"] is builtins.abs
            assert restored.absolute(-3) == 3
        assert module.__builtins__ is original_builtins
        assert module.absolute(-3) == 3
    finally:
        if registered:
            unregister_pickle_by_value(module)
    if registered:
        # Unregistered importable modules must still be serialized by reference.
        assert pickle.loads(dumps(module, protocol=protocol)) is module


@pytest.mark.parametrize("protocol", [2, 4, 5])
@pytest.mark.parametrize("as_function", [False, True])
def test_package_reference_cycle(protocol, as_function, tmp_path, monkeypatch):
    name = "cloudpickle_cycle_package"
    directory = tmp_path / name
    directory.mkdir()
    (directory / "__init__.py").write_text("answer = 42\nfrom . import child\n")
    (directory / "child.py").write_text(
        "import cloudpickle_cycle_package as parent\n"
        "def read():\n    return parent.answer\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    package = importlib.import_module(name)
    register_pickle_by_value(package)
    try:
        assert package.child.parent is package
        assert package.child.read() == 42

        def use_package():
            return package, package.child.read()

        obj = use_package if as_function else package
        for restored in (
            pickle.loads(dumps(obj, protocol=protocol)),
            subprocess_pickle_echo(obj, protocol=protocol),
        ):
            if as_function:
                restored, answer = restored()
                assert answer == 42
            assert restored is not package
            assert restored.child.parent is restored
            assert restored.child.read() == 42
        assert sys.modules[name] is package
    finally:
        unregister_pickle_by_value(package)
        sys.modules.pop(name + ".child", None)
        sys.modules.pop(name, None)
