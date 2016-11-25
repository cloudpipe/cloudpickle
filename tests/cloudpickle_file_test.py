import unittest
import tempfile
import os
import shutil
import pickle
import sys
try:
    from io import StringIO
except ImportError:
    # compat for Python 2.6
    from StringIO import StringIO

import pytest
from mock import patch, mock_open

import cloudpickle


class CloudPickleFileTests(unittest.TestCase):
    """In Cloudpickle, expected behaviour when pickling an opened file
    is to send its contents over the wire and seek to the same position."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tmpfilepath = os.path.join(self.tmpdir, 'testfile')
        self.teststring = u'Hello world!'

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_empty_file(self):
        # Empty file
        open(self.tmpfilepath, 'w').close()
        with open(self.tmpfilepath, 'r') as f:
            self.assertEquals('', pickle.loads(cloudpickle.dumps(f)).read())
        os.remove(self.tmpfilepath)

    def test_closed_file(self):
        # Write & close
        with open(self.tmpfilepath, 'w') as f:
            f.write(self.teststring)
        with pytest.raises(pickle.PicklingError) as excinfo:
            cloudpickle.dumps(f)
        assert "Cannot pickle closed files" in str(excinfo.value)
        os.remove(self.tmpfilepath)

    def test_r_mode(self):
        # Write & close
        with open(self.tmpfilepath, 'w') as f:
            f.write(self.teststring)
        # Open for reading
        with open(self.tmpfilepath, 'r') as f:
            new_f = pickle.loads(cloudpickle.dumps(f))
            self.assertEquals(self.teststring, new_f.read())
        os.remove(self.tmpfilepath)

    def test_w_mode(self):
        with open(self.tmpfilepath, 'w') as f:
            f.write(self.teststring)
            f.seek(0)
            self.assertRaises(pickle.PicklingError,
                              lambda: cloudpickle.dumps(f))
        os.remove(self.tmpfilepath)

    def test_plus_mode(self):
        # Write, then seek to 0
        with open(self.tmpfilepath, 'w+') as f:
            f.write(self.teststring)
            f.seek(0)
            new_f = pickle.loads(cloudpickle.dumps(f))
            self.assertEquals(self.teststring, new_f.read())
        os.remove(self.tmpfilepath)

    def test_seek(self):
        # Write, then seek to arbitrary position
        with open(self.tmpfilepath, 'w+') as f:
            f.write(self.teststring)
            f.seek(4)
            unpickled = pickle.loads(cloudpickle.dumps(f))
            # unpickled StringIO is at position 4
            self.assertEquals(4, unpickled.tell())
            self.assertEquals(self.teststring[4:], unpickled.read())
            # but unpickled StringIO also contained the start
            unpickled.seek(0)
            self.assertEquals(self.teststring, unpickled.read())
        os.remove(self.tmpfilepath)

    @pytest.mark.skipif(sys.version_info >= (3,),
                        reason="only works on Python 2.x")
    def test_temp_file(self):
        with tempfile.NamedTemporaryFile(mode='ab+') as fp:
            fp.write(self.teststring.encode('UTF-8'))
            fp.seek(0)
            f = fp.file
            # FIXME this doesn't work yet: cloudpickle.dumps(fp)
            newfile = pickle.loads(cloudpickle.dumps(f))
            self.assertEquals(self.teststring, newfile.read())

    def test_pickling_special_file_handles(self):
        # Warning: if you want to run your tests with nose, add -s option
        for out in sys.stdout, sys.stderr:  # Regression test for SPARK-3415
            self.assertEquals(out, pickle.loads(cloudpickle.dumps(out)))
        self.assertRaises(pickle.PicklingError,
                          lambda: cloudpickle.dumps(sys.stdin))

    def NOT_WORKING_test_tty(self):
        # FIXME: Mocking 'file' is not trivial... and fails for now
        from sys import version_info
        if version_info.major == 2:
            import __builtin__ as builtins  # pylint:disable=import-error
        else:
            import builtins  # pylint:disable=import-error

        with patch.object(builtins, 'open', mock_open(), create=True):
            with open('foo', 'w+') as handle:
                cloudpickle.dumps(handle)


if __name__ == '__main__':
    unittest.main()
