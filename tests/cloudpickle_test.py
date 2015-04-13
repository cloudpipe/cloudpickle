import unittest
import pickle
import sys

from operator import itemgetter, attrgetter
from StringIO import StringIO

from cloudpickle import cloudpickle

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

    def test_xrange_params(self):
        xr = xrange(1,100)
        start, step, length = cloudpickle.xrange_params(xr)
        self.assertEqual(1, start)
        self.assertEqual(1, step)
        self.assertEqual(99, length)

        xr = xrange(20)
        start, step, length = cloudpickle.xrange_params(xr)
        self.assertEqual(0, start)
        self.assertEqual(1, step)
        self.assertEqual(20, length)

        # Arbitrary steps
        xr = xrange(3,48,2)
        start, step, length = cloudpickle.xrange_params(xr)
        self.assertEqual(3, start)
        self.assertEqual(2, step)
        self.assertEqual(48/2-1, length)

        # Empty xrange
        xr = xrange(50,1,5)
        start, step, length = cloudpickle.xrange_params(xr)
        #self.assertEqual(50, start) # These currently are inferred
        #self.assertEqual(5, step)   # only by the length in Python2
        self.assertEqual(0, length)

        # Single element
        xr = xrange(42,43)
        start, step, length = cloudpickle.xrange_params(xr)
        self.assertEqual(1, length)
        self.assertEqual(42, start)
        self.assertEqual(1, step)


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

if __name__ == '__main__':
    unittest.main()
