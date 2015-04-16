import sys
import os
from subprocess import Popen
from subprocess import PIPE

from cloudpickle import dumps
from pickle import loads

try:
    from suprocess import TimeoutExpired
    timeout_supported = True
except ImportError:
    # no support for timeout in Python 2
    class TimeoutExpired(Exception):
        pass
    timeout_supported = False


def subprocess_pickle_echo(input_data):
    """Echo function with a child Python process

    Pickle the input data into a buffer, send it to a subprocess via
    stdin, expect the subprocess to unpickle, re-pickle that data back
    and send it back to the parent process via stdout for final unpickling.

    >>> subprocess_pickle_echo([1, 'a', None])
    [1, 'a', None]

    """
    pickled_input_data = dumps(input_data)
    cmd = [sys.executable, __file__]
    cwd = os.getcwd()
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd)
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


def pickle_echo(stream_in=None, stream_out=None):
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
    stream_out.write(dumps(unpickled_content))
    stream_out.close()


if __name__ == '__main__':
    pickle_echo()
