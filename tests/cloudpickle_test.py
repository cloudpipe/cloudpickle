import unittest
import pytest
import pickle
import sys

from operator import itemgetter, attrgetter

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import cloudpickle


def pickle_depickle(obj):
    """Helper function to test whether object pickled with cloudpickle can be
    depickled with pickle
    """
    return pickle.loads(cloudpickle.dumps(obj))


class CloudPicklerTest(unittest.TestCase):
    def setUp(self):
        self.file_obj = StringIO()
        self.cloudpickler = cloudpickle.CloudPickler(self.file_obj, 2)


class CloudPickleTest(unittest.TestCase):

    def test_itemgetter(self):
        d = range(10)
        getter = itemgetter(1)

        getter2 = pickle.loads(cloudpickle.dumps(getter))
        self.assertEqual(getter(d), getter2(d))

        getter = itemgetter(0, 3)
        getter2 = pickle.loads(cloudpickle.dumps(getter))
        self.assertEqual(getter(d), getter2(d))

    def test_attrgetter(self):
        class C(object):
            def __getattr__(self, item):
                return item
        d = C()
        getter = attrgetter("a")
        getter2 = pickle.loads(cloudpickle.dumps(getter))
        self.assertEqual(getter(d), getter2(d))
        getter = attrgetter("a", "b")
        getter2 = pickle.loads(cloudpickle.dumps(getter))
        self.assertEqual(getter(d), getter2(d))

        d.e = C()
        getter = attrgetter("e.a")
        getter2 = pickle.loads(cloudpickle.dumps(getter))
        self.assertEqual(getter(d), getter2(d))
        getter = attrgetter("e.a", "e.b")
        getter2 = pickle.loads(cloudpickle.dumps(getter))
        self.assertEqual(getter(d), getter2(d))

    def test_pickling_xrange(self):
        xr1 = xrange(1,100, 3)
        xr2 = pickle.loads(cloudpickle.dumps(xr1))

        # Can't just `self.assertEquals(xr1, xr2)` because it compares
        # the objects
        for a,b in zip(xr1, xr2):
            self.assertEquals(a,b)

    # Regression test for SPARK-3415
    def test_pickling_file_handles(self):
        out1 = sys.stderr
        out2 = pickle.loads(cloudpickle.dumps(out1))
        self.assertEquals(out1, out2)

    def test_func_globals(self):
        class Unpicklable(object):
            def __reduce__(self):
                raise Exception("not picklable")

        global exit
        exit = Unpicklable()

        self.assertRaises(Exception, lambda: cloudpickle.dumps(exit))

        def foo():
            sys.exit(0)

        self.assertTrue("exit" in foo.func_code.co_names)
        cloudpickle.dumps(foo)

    def test_buffer(self):
        try:
            buffer_obj = buffer("Hello")
            self.assertEqual(pickle_depickle(buffer_obj), str(buffer_obj))
            buffer_obj = buffer("Hello", 2, 3)
            self.assertEqual(pickle_depickle(buffer_obj), str(buffer_obj))
        except NameError:  # Python 3 does no longer support buffers
            pass

    def test_lambda(self):
        self.assertEqual(pickle_depickle(lambda: 1)(), 1)

    def test_save_unsupported(self):
        sio = StringIO()
        pickler = cloudpickle.CloudPickler(sio, 2)

        with pytest.raises(pickle.PicklingError) as excinfo:
            pickler.save_unsupported("test")

        assert "Cannot pickle objects of type" in str(excinfo.value)


if __name__ == '__main__':
    unittest.main()
