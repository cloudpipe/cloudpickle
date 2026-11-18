import io
import os
import pickle
import subprocess
import sys
import textwrap
import uuid
from unittest import mock

import pytest

import cloudpickle

pytest.importorskip("Cython")

CYTHON_SOURCE = """
def plus_two(value):
    return value + 2

cpdef int triple(int value):
    return value * 3

class Plain:
    def __init__(self, value=4):
        self.value = value

    def run(self, value):
        return self.value + value

cdef class Native:
    cdef public int value

    def __init__(self, value=5):
        self.value = value

    def run(self, value):
        return self.value + value

alias = plus_two
counter = 0

def increment():
    global counter
    counter += 1
    return counter
"""


@pytest.fixture(scope="module", params=["pyximport", "cython_magic"])
def cython_module(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(request.param)
    name = "cloudpickle_cython_" + uuid.uuid4().hex
    if request.param == "pyximport":
        import pyximport

        (directory / (name + ".pyx")).write_text(CYTHON_SOURCE)
        importers = pyximport.install(
            build_dir=str(directory / "build"), language_level=3
        )
        sys.path.insert(0, str(directory))
        try:
            module = __import__(name)
        finally:
            sys.path.remove(str(directory))
            pyximport.uninstall(*importers)
    else:
        from Cython.Build.IpythonMagic import CythonMagics
        from IPython.core.interactiveshell import InteractiveShell

        shell = InteractiveShell()
        magic = CythonMagics(shell)
        with mock.patch(
            "Cython.Build.IpythonMagic.get_ipython_cache_dir",
            return_value=str(directory),
        ):
            magic.cython("--name " + name, CYTHON_SOURCE)
        module = sys.modules[shell.user_ns["plus_two"].__module__]
    yield module
    sys.modules.pop(module.__name__, None)


def run_in_fresh_process(payload, expression, setup=""):
    program = textwrap.dedent(
        """
        import pickle
        import sys
        """
    )
    program += setup + "\n"
    program += "objects = pickle.loads(sys.stdin.buffer.read())\n"
    program += "result = " + expression + "\n"
    program += "sys.stdout.buffer.write(pickle.dumps(result))\n"
    process = subprocess.run(
        [sys.executable, "-c", program],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=True,
    )
    return pickle.loads(process.stdout)


@pytest.mark.parametrize("protocol", [2, 4, 5])
def test_shared_cython_functions_and_classes(cython_module, protocol):
    module = cython_module
    objects = (
        module.plus_two,
        module.triple,
        module.Plain,
        module.Native,
        module.alias,
    )
    payload = cloudpickle.dumps(
        objects,
        protocol=protocol,
        assume_shared_dynamic_cython_module_folder=True,
    )
    assert run_in_fresh_process(
        payload,
        "(objects[0](5), objects[1](5), objects[2]().run(5), "
        "objects[3]().run(5), objects[0] is objects[4], "
        "'pyximport' in sys.modules, 'Cython' in sys.modules)",
    ) == (7, 15, 9, 10, True, False, False)


def test_shared_cython_module_and_python_wrapper(cython_module):
    module = cython_module

    def wrapped(value):
        return module.plus_two(value)

    payload = cloudpickle.dumps(
        (module, wrapped), assume_shared_dynamic_cython_module_folder=True
    )
    assert run_in_fresh_process(payload, "(objects[0].triple(4), objects[1](4))") == (
        12,
        6,
    )


def test_shared_cython_instances_and_bound_methods(cython_module):
    module = cython_module
    instance = module.Native(9)
    plain = module.Plain(7)
    payload = cloudpickle.dumps(
        (instance, instance.run, module.Native, plain, plain.run, module.Plain),
        assume_shared_dynamic_cython_module_folder=True,
    )
    assert run_in_fresh_process(
        payload,
        "(objects[0].value, objects[1](1), type(objects[0]) is objects[2], "
        "objects[3].value, objects[4](1), type(objects[3]) is objects[5])",
    ) == (9, 10, True, 7, 8, True)


def test_shared_cython_is_opt_in(cython_module):
    default = cloudpickle.dumps(cython_module.plus_two)
    disabled = cloudpickle.dumps(
        cython_module.plus_two, assume_shared_dynamic_cython_module_folder=False
    )
    assert default == disabled
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_in_fresh_process(default, "objects(1)")
    assert b"ModuleNotFoundError" in raised.value.stderr


def test_shared_cython_pickler_and_dump(cython_module):
    outputs = []
    for constructor in ("pickler", "dump"):
        stream = io.BytesIO()
        if constructor == "pickler":
            pickler = cloudpickle.Pickler(
                stream, assume_shared_dynamic_cython_module_folder=True
            )
            pickler.dump(cython_module.plus_two)
        else:
            cloudpickle.dump(
                cython_module.plus_two,
                stream,
                assume_shared_dynamic_cython_module_folder=True,
            )
        outputs.append(run_in_fresh_process(stream.getvalue(), "objects(3)"))
    assert outputs == [5, 5]


def test_shared_cython_preserves_existing_module(cython_module):
    payload = cloudpickle.dumps(
        (cython_module, cython_module.plus_two),
        assume_shared_dynamic_cython_module_folder=True,
    )
    module, function = pickle.loads(payload)
    assert module is cython_module
    assert function is cython_module.plus_two


def test_shared_cython_rejects_conflicting_module(cython_module):
    payload = cloudpickle.dumps(
        cython_module.plus_two, assume_shared_dynamic_cython_module_folder=True
    )
    setup = (
        "import types\n"
        f"conflict = types.ModuleType({cython_module.__name__!r})\n"
        "conflict.__file__ = 'different-extension.so'\n"
        "sys.modules[conflict.__name__] = conflict\n"
    )
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_in_fresh_process(payload, "objects(1)", setup)
    assert b"ImportError" in raised.value.stderr
    assert b"already loaded" in raised.value.stderr


def test_shared_cython_keeps_importable_extensions_by_reference():
    import math

    objects = (math, math.sin)
    assert cloudpickle.dumps(
        objects, assume_shared_dynamic_cython_module_folder=True
    ) == cloudpickle.dumps(objects)


def test_shared_cython_reuses_module_state_across_loads(cython_module):
    payload = cloudpickle.dumps(
        cython_module.increment, assume_shared_dynamic_cython_module_folder=True
    )
    assert run_in_fresh_process(
        payload,
        "(objects(), pickle.loads(payload)(), " "objects is pickle.loads(payload))",
        f"payload = {payload!r}",
    ) == (1, 2, True)


def test_shared_cython_missing_binary_has_no_import_side_effects(
    cython_module, tmp_path
):
    # Model a shared filename that is absent on the receiver without renaming
    # a loaded DLL, which Windows does not allow.
    missing_path = str(tmp_path / os.path.basename(cython_module.__file__))
    with mock.patch.object(cython_module, "__file__", missing_path):
        payload = cloudpickle.dumps(
            cython_module.plus_two, assume_shared_dynamic_cython_module_folder=True
        )
    program = (
        "import pickle,sys\n"
        "before = tuple(sys.meta_path)\n"
        "try:\n"
        "    pickle.loads(sys.stdin.buffer.read())\n"
        "except ImportError:\n"
        f"    assert {cython_module.__name__!r} not in sys.modules\n"
        "    assert tuple(sys.meta_path) == before\n"
        "else:\n"
        "    raise AssertionError('missing extension was loaded')\n"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        input=payload,
        capture_output=True,
        timeout=60,
        check=True,
    )


def test_shared_cython_blocked_module_stays_blocked(cython_module):
    payload = cloudpickle.dumps(
        cython_module.plus_two, assume_shared_dynamic_cython_module_folder=True
    )
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_in_fresh_process(
            payload,
            "objects(1)",
            f"sys.modules[{cython_module.__name__!r}] = None",
        )
    assert b"ModuleNotFoundError" in raised.value.stderr


def test_shared_cython_respects_module_registration(cython_module):
    cloudpickle.register_pickle_by_value(cython_module)
    try:
        from cloudpickle.cloudpickle import _extension_module_path

        assert _extension_module_path(cython_module) is None
    finally:
        cloudpickle.unregister_pickle_by_value(cython_module)


def test_shared_cython_unhashable_python_module():
    import types

    class Module(types.ModuleType):
        __hash__ = None

    module = Module("cloudpickle_unhashable_python_module")
    exec("def function(): return 42", module.__dict__)
    sys.modules[module.__name__] = module
    try:
        assert cloudpickle.dumps(
            module.function, assume_shared_dynamic_cython_module_folder=True
        ) == cloudpickle.dumps(module.function)
    finally:
        del sys.modules[module.__name__]


def test_shared_cython_success_cleans_import_hooks(cython_module):
    payload = cloudpickle.dumps(
        cython_module.plus_two, assume_shared_dynamic_cython_module_folder=True
    )
    assert run_in_fresh_process(
        payload,
        "(objects(1), tuple(sys.meta_path) == before)",
        "before = tuple(sys.meta_path)",
    ) == (3, True)


def test_shared_cython_reexporting_parent(tmp_path):
    import pyximport

    package_name = "cloudpickle_pkg_" + uuid.uuid4().hex
    package = tmp_path / package_name
    package.mkdir()
    (package / "__init__.py").write_text("from . import native\nbefore = native\n")
    (package / "native.pyx").write_text(CYTHON_SOURCE)
    importers = pyximport.install(build_dir=str(tmp_path / "build"), language_level=3)
    sys.path.insert(0, str(tmp_path))
    try:
        parent = __import__(package_name)
    finally:
        sys.path.remove(str(tmp_path))
        pyximport.uninstall(*importers)
    module = parent.native
    try:
        payload = cloudpickle.dumps(
            (module, module.plus_two, module.Native),
            assume_shared_dynamic_cython_module_folder=True,
        )
        assert run_in_fresh_process(
            payload,
            f"(objects[1](1), objects[2]().run(1), "
            f"sys.modules[{package_name!r}].before is objects[0], "
            "objects[2] is objects[0].Native)",
            f"sys.path.insert(0, {str(tmp_path)!r})",
        ) == (3, 6, True, True)
    finally:
        sys.modules.pop(module.__name__, None)
        sys.modules.pop(package_name, None)


def test_shared_cython_initialization_coordinates_imports(tmp_path):
    import pyximport

    name = "cloudpickle_initializing_" + uuid.uuid4().hex
    source = """
import os
import time
from pathlib import Path
started = os.environ.get("CLOUDPICKLE_CYTHON_STARTED")
if started:
    Path(started).write_text("started")
    while not Path(os.environ["CLOUDPICKLE_CYTHON_RELEASE"]).exists():
        time.sleep(0.005)
ready = True
def function():
    return ready
"""
    (tmp_path / (name + ".pyx")).write_text(source)
    importers = pyximport.install(build_dir=str(tmp_path / "build"), language_level=3)
    sys.path.insert(0, str(tmp_path))
    try:
        module = __import__(name)
    finally:
        sys.path.remove(str(tmp_path))
        pyximport.uninstall(*importers)
    try:
        payload = cloudpickle.dumps(
            module, assume_shared_dynamic_cython_module_folder=True
        )
        program = textwrap.dedent(
            """
            import importlib
            import os
            import pickle
            from pathlib import Path
            import sys
            import threading
            import time
            from concurrent.futures import ThreadPoolExecutor

            name, directory = sys.argv[1:]
            started = Path(directory) / "started"
            release = Path(directory) / "release"
            os.environ["CLOUDPICKLE_CYTHON_STARTED"] = str(started)
            os.environ["CLOUDPICKLE_CYTHON_RELEASE"] = str(release)
            payload = sys.stdin.buffer.read()
            before = tuple(sys.meta_path)
            entered = threading.Event()
            done = threading.Event()

            def ordinary_import():
                entered.set()
                imported = importlib.import_module(name)
                done.set()
                return imported

            with ThreadPoolExecutor(max_workers=2) as pool:
                restored = pool.submit(pickle.loads, payload)
                deadline = time.monotonic() + 10
                while not started.exists():
                    if restored.done():
                        restored.result()
                    assert time.monotonic() < deadline
                    time.sleep(0.005)
                imported = pool.submit(ordinary_import)
                try:
                    assert entered.wait(10)
                    assert not done.wait(0.1)
                finally:
                    release.touch()
                assert restored.result(timeout=10) is imported.result(timeout=10)
                assert restored.result().ready
            assert tuple(sys.meta_path) == before
            """
        )
        subprocess.run(
            [sys.executable, "-c", program, name, str(tmp_path)],
            input=payload,
            capture_output=True,
            timeout=60,
            check=True,
        )
    finally:
        sys.modules.pop(name, None)


def test_shared_cython_failed_initialization_can_retry(tmp_path):
    import pyximport

    name = "cloudpickle_init_failure_" + uuid.uuid4().hex
    source = """
import os
if os.environ.get("CLOUDPICKLE_CYTHON_FAIL"):
    raise RuntimeError("extension initialization failed")
def function():
    return 42
"""
    (tmp_path / (name + ".pyx")).write_text(source)
    importers = pyximport.install(build_dir=str(tmp_path / "build"), language_level=3)
    sys.path.insert(0, str(tmp_path))
    try:
        module = __import__(name)
    finally:
        sys.path.remove(str(tmp_path))
        pyximport.uninstall(*importers)
    try:
        payload = cloudpickle.dumps(
            module.function, assume_shared_dynamic_cython_module_folder=True
        )
        program = textwrap.dedent(
            """
            import os
            import pickle
            import sys
            name = sys.argv[1]
            payload = sys.stdin.buffer.read()
            before = tuple(sys.meta_path)
            os.environ["CLOUDPICKLE_CYTHON_FAIL"] = "1"
            try:
                pickle.loads(payload)
            except RuntimeError as error:
                assert str(error) == "extension initialization failed"
            else:
                raise AssertionError("expected module initialization to fail")
            assert name not in sys.modules
            assert tuple(sys.meta_path) == before
            del os.environ["CLOUDPICKLE_CYTHON_FAIL"]
            assert pickle.loads(payload)() == 42
            assert tuple(sys.meta_path) == before
            """
        )
        subprocess.run(
            [sys.executable, "-c", program, name],
            input=payload,
            capture_output=True,
            timeout=60,
            check=True,
        )
    finally:
        sys.modules.pop(name, None)


def test_shared_cython_importable_compiled_module_uses_reference(cython_module):
    directory = os.path.dirname(cython_module.__file__)
    sys.path.insert(0, directory)
    try:
        objects = (cython_module, cython_module.plus_two, cython_module.Native)
        assert cloudpickle.dumps(
            objects, assume_shared_dynamic_cython_module_folder=True
        ) == cloudpickle.dumps(objects)
    finally:
        sys.path.remove(directory)
