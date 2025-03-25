import pytest
import textwrap

from .testutils import check_deterministic_pickle

try:
    import ipykernel
except ImportError:
    pytest.skip("ipykernel is not installed", allow_module_level=True)


def run_in_notebook(code):
    km = ipykernel.connect.jupyter_client.KernelManager()
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    status, output, err = "run", None, None
    try:
        assert km.is_alive() and kc.is_alive()
        kc.wait_for_ready()
        idx = kc.execute(code)
        running = True
        while running:
            res = kc.iopub_channel.get_msg(timeout=None)
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
    if status != "error":
        status = "ok" if not running else "exec_error"
    return status, output, err


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
