import sys
import os
import os.path as op
import tempfile
import base64
from subprocess import Popen, check_output, PIPE, STDOUT, CalledProcessError
from cloudpickle import dumps
from pickle import loads

TIMEOUT = 60
try:
    from subprocess import TimeoutExpired
    timeout_supported = True
except ImportError:
    # no support for timeout in Python 2
    class TimeoutExpired(Exception):
        pass
    timeout_supported = False


TEST_GLOBALS = "a test value"


def make_local_function():
    def g(x):
        # this function checks that the globals are correctly handled and that
        # the builtins are available
        assert TEST_GLOBALS == "a test value"
        return sum(range(10))

    return g


def _make_cwd_env():
    """Helper to prepare environment for the child processes"""
    cloudpickle_repo_folder = op.normpath(
        op.join(op.dirname(__file__), '..'))
    env = os.environ.copy()
    pythonpath = "{src}{sep}tests{pathsep}{src}".format(
        src=cloudpickle_repo_folder, sep=os.sep, pathsep=os.pathsep)
    env['PYTHONPATH'] = pythonpath
    return cloudpickle_repo_folder, env


def subprocess_pickle_echo(input_data, protocol=None, timeout=TIMEOUT):
    """Echo function with a child Python process

    Pickle the input data into a buffer, send it to a subprocess via
    stdin, expect the subprocess to unpickle, re-pickle that data back
    and send it back to the parent process via stdout for final unpickling.

    >>> subprocess_pickle_echo([1, 'a', None])
    [1, 'a', None]

    """
    pickled_input_data = dumps(input_data, protocol=protocol)
    # Under Windows + Python 2.7, subprocess / communicate truncate the data
    # on some specific bytes. To avoid this issue, let's use the pure ASCII
    # Base32 encoding to encapsulate the pickle message sent to the child
    # process.
    pickled_b32 = base64.b32encode(pickled_input_data)

    # run then pickle_echo(protocol=protocol) in __main__:
    cmd = [sys.executable, __file__, "--protocol", str(protocol)]
    cwd, env = _make_cwd_env()
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd, env=env,
                 bufsize=4096)
    try:
        comm_kwargs = {}
        if timeout_supported:
            comm_kwargs['timeout'] = timeout
        out, err = proc.communicate(pickled_b32, **comm_kwargs)
        if proc.returncode != 0 or len(err):
            message = "Subprocess returned %d: " % proc.returncode
            message += err.decode('utf-8')
            raise RuntimeError(message)
        return loads(base64.b32decode(out))
    except TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        message = u"\n".join([out.decode('utf-8'), err.decode('utf-8')])
        raise RuntimeError(message)


def _read_all_bytes(stream_in, chunk_size=4096):
    all_data = b""
    while True:
        data = stream_in.read(chunk_size)
        all_data += data
        if len(data) < chunk_size:
            break
    return all_data


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

    input_bytes = base64.b32decode(_read_all_bytes(stream_in))
    stream_in.close()
    obj = loads(input_bytes)
    repickled_bytes = dumps(obj, protocol=protocol)
    stream_out.write(base64.b32encode(repickled_bytes))
    stream_out.close()


def assert_run_python_script(source_code, timeout=TIMEOUT):
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
        cwd, env = _make_cwd_env()
        kwargs = {
            'cwd': cwd,
            'stderr': STDOUT,
            'env': env,
        }
        # If coverage is running, pass the config file to the subprocess
        coverage_rc = os.environ.get("COVERAGE_PROCESS_START")
        if coverage_rc:
            kwargs['env']['COVERAGE_PROCESS_START'] = coverage_rc
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


if __name__ == '__main__':
    protocol = int(sys.argv[sys.argv.index('--protocol') + 1])
    pickle_echo(protocol=protocol)
