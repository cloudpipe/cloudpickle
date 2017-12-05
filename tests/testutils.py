import sys
import os
import os.path as op
import tempfile
from subprocess import Popen, check_output, PIPE, STDOUT, CalledProcessError
import gc
import multiprocessing as mp

from cloudpickle import dumps
from pickle import loads
import pytest

try:
    from suprocess import TimeoutExpired
    timeout_supported = True
except ImportError:
    # no support for timeout in Python 2
    class TimeoutExpired(Exception):
        pass
    timeout_supported = False


def subprocess_pickle_echo(input_data, protocol=None):
    """Echo function with a child Python process

    Pickle the input data into a buffer, send it to a subprocess via
    stdin, expect the subprocess to unpickle, re-pickle that data back
    and send it back to the parent process via stdout for final unpickling.

    >>> subprocess_pickle_echo([1, 'a', None])
    [1, 'a', None]

    """
    pickled_input_data = dumps(input_data, protocol=protocol)
    cmd = [sys.executable, __file__]  # run then pickle_echo() in __main__
    cloudpickle_repo_folder = op.normpath(
        op.join(op.dirname(__file__), '..'))
    cwd = cloudpickle_repo_folder
    pythonpath = "{src}/tests:{src}".format(src=cloudpickle_repo_folder)
    env = {'PYTHONPATH': pythonpath}
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd, env=env)
    try:
        comm_kwargs = {}
        if timeout_supported:
            comm_kwargs['timeout'] = 5
        out, err = proc.communicate(pickled_input_data, **comm_kwargs)
        if proc.returncode != 0 or len(err):
            message = "Subprocess returned %d: " % proc.returncode
            message += err.decode('utf-8')
            raise RuntimeError(message)
        return loads(out)
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        message = u"\n".join([out.decode('utf-8'), err.decode('utf-8')])
        raise RuntimeError(message)


def pickle_echo(stream_in=None, stream_out=None, protocol=None):
    """Read a pickle from stdin and pickle it back to stdout"""
    if stream_in is None:
        stream_in = sys.stdin
    if stream_out is None:
        stream_out = sys.stdout

    # Force the use of bytes streams under Python 3
    if hasattr(stream_in, 'buffer'):
        stream_in = stream_in.buffer
    if hasattr(stream_out, 'buffer'):
        stream_out = stream_out.buffer

    input_bytes = stream_in.read()
    stream_in.close()
    unpickled_content = loads(input_bytes)
    stream_out.write(dumps(unpickled_content, protocol=protocol))
    stream_out.close()


def assert_run_python_script(source_code, timeout=5):
    """Utility to help check pickleability of objects defined in __main__

    The script provided in the source code should return 0 and not print
    anything on stderr or stdout.
    """
    fd, source_file = tempfile.mkstemp(suffix='_src_test_cloudpickle.py')
    os.close(fd)
    try:
        with open(source_file, 'wb') as f:
            f.write(source_code.encode('utf-8'))
        cmd = [sys.executable, source_file]
        cloudpickle_repo_folder = op.normpath(
            op.join(op.dirname(__file__), '..'))
        pythonpath = "{src}/tests:{src}".format(src=cloudpickle_repo_folder)
        kwargs = {
            'cwd': cloudpickle_repo_folder,
            'stderr': STDOUT,
            'env': {'PYTHONPATH': pythonpath},
        }
        if timeout_supported:
            kwargs['timeout'] = timeout
        try:
            try:
                out = check_output(cmd, **kwargs)
            except CalledProcessError as e:
                raise RuntimeError(u"script errored with output:\n%s"
                                   % e.output.decode('utf-8'))
            if out != b"":
                raise AssertionError(out.decode('utf-8'))
        except TimeoutExpired as e:
            raise RuntimeError(u"script timeout, output so far:\n%s"
                               % e.output.decode('utf-8'))
    finally:
        os.unlink(source_file)


def monitor_worker(pid, queue, stop_event, delay=0.05):
    import psutil
    p = psutil.Process(pid)
    peak = 0

    def make_measurement(peak):
        mem = p.memory_info().rss
        if mem > peak:
            peak = mem
        return peak

    # Make measurements every 'delay' seconds until we receive the stop event:
    while not stop_event.wait(timeout=delay):
        peak = make_measurement(peak)

    # Make one last measurement in case memory has increased just before
    # receiving the stop event:
    peak = make_measurement(peak)
    queue.put(peak)


class PeakMemoryMonitor:

    def __enter__(self):
        psutil = pytest.importorskip("psutil")
        pid = os.getpid()
        gc.collect()
        self.base_mem = psutil.Process(pid).memory_info().rss
        self.queue = q = mp.Queue()
        self.stop_event = e = mp.Event()
        self.worker = mp.Process(target=monitor_worker, args=(pid, q, e))
        self.worker.start()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.stop_event.set()
        if exc_type is not None:
            self.worker.terminate()
            return False
        else:
            self.peak_mem = self.queue.get()
            return True


if __name__ == '__main__':
    pickle_echo()
