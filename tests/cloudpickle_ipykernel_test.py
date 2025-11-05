import sys
import time
import pytest
import platform
import textwrap
from queue import Empty

from .testutils import check_deterministic_pickle

if sys.platform == "win32":
    if sys.version_info < (3, 11):
        pytest.skip(
            "ipykernel requires Python 3.11 or later",
            allow_module_level=True
        )
ipykernel = pytest.importorskip("ipykernel")


def run_in_notebook(code, timeout=10):

    km = ipykernel.connect.jupyter_client.KernelManager()
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    status, output, err = "kernel_started", None, None
    try:
        assert km.is_alive() and kc.is_alive()
        kc.wait_for_ready()
        idx = kc.execute(code)
        running = True
        while running:
            try:
                res = kc.iopub_channel.get_msg(timeout=timeout)
            except Empty:
                status = "timeout"
                break
            if res['parent_header'].get('msg_id') != idx:
                continue
            content = res['content']
            if content.get("name", "state") == "stdout":
                output = content['text']
            if "traceback" in content:
                err = "\n".join(content['traceback'])
                status = "error"
            running = res['content'].get('execution_state', None) != "idle"
    finally:
        kc.shutdown()
        kc.stop_channels()
        km.shutdown_kernel(now=True, restart=False)
        assert not km.is_alive()
    if status not in ["error", "timeout"]:
        status = "ok" if not running else "exec_error"
    return status, output, err


@pytest.mark.skipif(
    platform.python_implementation() == "PyPy",
    reason="Skip PyPy because tests are too slow",
)
@pytest.mark.parametrize("code, expected", [
    ("1 + 1", "ok"),
    ("raise ValueError('This is a test error')", "error"),
    ("import time; time.sleep(100)", "timeout")

])
def test_run_in_notebook(code, expected):
    code = textwrap.dedent(code)

    t_start = time.time()
    status, output, err = run_in_notebook(code, timeout=1)
    duration = time.time() - t_start
    assert status == expected, (
        f"Unexpected status: {status}, output: {output}, err: {err}, duration: {duration}"
    )
    assert duration < 10, "Timeout not enforced properly"
    if expected == "error":
        assert "This is a test error" in err


def test_deterministic_payload_for_dynamic_func_in_notebook():
    code = textwrap.dedent("""
        import cloudpickle

        MY_PI = 3.1415

        def get_pi():
            return MY_PI

        print(cloudpickle.dumps(get_pi))
    """)

    status, output, err = run_in_notebook(code)
    assert status == "ok"
    payload = eval(output.strip(), {})

    status, output, err = run_in_notebook(code)
    assert status == "ok"
    payload2 = eval(output.strip(), {})

    check_deterministic_pickle(payload, payload2)
