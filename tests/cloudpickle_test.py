from __future__ import division
import unittest
import pytest
import pickle
import sys
import functools

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

        func_code = getattr(foo, '__code__', None)
        if func_code is None: # PY2 backwards compatibility
            func_code = foo.func_code

        self.assertTrue("exit" in func_code.co_names)
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

    def test_nested_lambdas(self):
        a, b = 1, 2
        f1 = lambda x: x + a
        f2 = lambda x: f1(x) // b
        self.assertEqual(pickle_depickle(f2)(1), 1)

    def test_locally_defined_function_and_class(self):
        LOCAL_CONSTANT = 42

        def some_function(x, y):
            return (x + y) / LOCAL_CONSTANT

        # pickle the function definition
        self.assertEqual(pickle_depickle(some_function)(41, 1), 1)
        self.assertEqual(pickle_depickle(some_function)(81, 3), 2)

        hidden_constant = lambda: LOCAL_CONSTANT

        class SomeClass(object):
            """Overly complicated class with nested references to symbols"""
            def __init__(self, value):
                self.value = value

            def one(self):
                return LOCAL_CONSTANT / hidden_constant()

            def some_method(self, x):
                return self.one() + some_function(x, 1) + self.value

        # pickle the class definition
        self.assertEqual(pickle_depickle(SomeClass)(1).one(), 1)
        self.assertEqual(pickle_depickle(SomeClass)(5).some_method(41), 7)

        # pickle the class instances
        self.assertEqual(pickle_depickle(SomeClass(1)).one(), 1)
        self.assertEqual(pickle_depickle(SomeClass(5)).some_method(41), 7)

        # pickle the method instances
        self.assertEqual(pickle_depickle(SomeClass(1).one)(), 1)
        self.assertEqual(pickle_depickle(SomeClass(5).some_method)(41), 7)

    def test_partial(self):
        partial_obj = functools.partial(min, 1)
        self.assertEqual(pickle_depickle(partial_obj)(4), 1)

    def test_save_unsupported(self):
        sio = StringIO()
        pickler = cloudpickle.CloudPickler(sio, 2)

        with pytest.raises(pickle.PicklingError) as excinfo:
            pickler.save_unsupported("test")

        assert "Cannot pickle objects of type" in str(excinfo.value)


if __name__ == '__main__':
    unittest.main()
