"""Check the distributions used by downstream package maintainers."""

import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest
from flit_core import buildapi


@pytest.fixture
def distributions(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    # Copy build inputs only, not virtual environments such as .tox or .venv.
    for name in (
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "dev-requirements.txt",
        "tox.ini",
    ):
        shutil.copy2(project / name, source / name)
    for name in ("cloudpickle", "tests"):
        shutil.copytree(
            project / name,
            source / name,
            ignore=shutil.ignore_patterns(
                ".pytest_cache", "__pycache__", "*.py[co]", "*.egg-info", "build"
            ),
        )
    # Installing the test helper and running tests must not pollute an sdist.
    generated = [
        "tests/__pycache__/cloudpickle_test.cpython-312.pyc",
        "tests/testutils.pyc",
        "tests/cloudpickle_testpkg/cloudpickle_testpkg.egg-info/PKG-INFO",
        "tests/cloudpickle_testpkg/build/lib/_cloudpickle_testpkg/__init__.py",
    ]
    for name in generated:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated file")

    dist = tmp_path / "dist"
    dist.mkdir()
    monkeypatch.chdir(source)
    sdist = dist / buildapi.build_sdist(str(dist))
    with tarfile.open(sdist) as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames()}
        # This archive was built locally from the test's own source snapshot.
        options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        archive.extractall(tmp_path / "unpacked", **options)
    unpacked = next((tmp_path / "unpacked").iterdir())
    monkeypatch.chdir(unpacked)
    wheel = dist / buildapi.build_wheel(str(dist))
    return source, names, generated, wheel


def test_sdist_contains_test_suite(distributions):
    _, names, generated, _ = distributions
    required = {
        "dev-requirements.txt",
        "tox.ini",
        "tests/__init__.py",
        "tests/cloudpickle_test.py",
        "tests/cloudpickle_file_test.py",
        "tests/cloudpickle_ipykernel_test.py",
        "tests/test_backward_compat.py",
        "tests/test_packaging.py",
        "tests/testutils.py",
        "tests/mock_local_folder/subfolder/submod.py",
        "tests/cloudpickle_testpkg/setup.py",
        "tests/cloudpickle_testpkg/_cloudpickle_testpkg/mod.py",
        "tests/old_pickles/cpython_311/simple_func.pkl",
    }
    assert required <= names, required - names
    assert not set(generated) & names


def test_wheel_from_sdist_contains_only_runtime_package(distributions):
    source, _, _, wheel = distributions
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert all(
            name.startswith("cloudpickle/") or ".dist-info/" in name for name in names
        )
        for name in names:
            if name.startswith("cloudpickle/"):
                assert archive.read(name) == (source / name).read_bytes()
