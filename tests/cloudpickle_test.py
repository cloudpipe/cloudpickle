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
