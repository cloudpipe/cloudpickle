from __future__ import division

import abc
import collections
import base64
import functools
import imp
from io import BytesIO
import itertools
import logging
from operator import itemgetter, attrgetter
import pickle
import platform
import random
import subprocess
import sys
import textwrap
import unittest
import weakref

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import pytest

try:
    # try importing numpy and scipy. These are not hard dependencies and
    # tests should be skipped if these modules are not available
    import numpy as np
    import scipy.special as spp
except ImportError:
    np = None
    spp = None

try:
    # Ditto for Tornado
    import tornado
except ImportError:
    tornado = None

import cloudpickle
from cloudpickle.cloudpickle import _find_module, _make_empty_cell, cell_set

from .testutils import subprocess_pickle_echo


HAVE_WEAKSET = hasattr(weakref, 'WeakSet')


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

        getter2 = pickle_depickle(getter)
        self.assertEqual(getter(d), getter2(d))

        getter = itemgetter(0, 3)
        getter2 = pickle_depickle(getter)
        self.assertEqual(getter(d), getter2(d))

    def test_attrgetter(self):
        class C(object):
            def __getattr__(self, item):
                return item
        d = C()
        getter = attrgetter("a")
        getter2 = pickle_depickle(getter)
        self.assertEqual(getter(d), getter2(d))
        getter = attrgetter("a", "b")
        getter2 = pickle_depickle(getter)
        self.assertEqual(getter(d), getter2(d))

        d.e = C()
        getter = attrgetter("e.a")
        getter2 = pickle_depickle(getter)
        self.assertEqual(getter(d), getter2(d))
        getter = attrgetter("e.a", "e.b")
        getter2 = pickle_depickle(getter)
        self.assertEqual(getter(d), getter2(d))

    # Regression test for SPARK-3415
    def test_pickling_file_handles(self):
        out1 = sys.stderr
        out2 = pickle.loads(cloudpickle.dumps(out1))
        self.assertEqual(out1, out2)

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
        if func_code is None:  # PY2 backwards compatibility
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

    def test_recursive_closure(self):
        def f1():
            def g():
                return g
            return g

        def f2(base):
            def g(n):
                return base if n <= 1 else n * g(n - 1)
            return g

        g1 = pickle_depickle(f1())
        self.assertEqual(g1(), g1)

        g2 = pickle_depickle(f2(2))
        self.assertEqual(g2(5), 240)

    def test_closure_none_is_preserved(self):
        def f():
            """a function with no closure cells
            """

        self.assertTrue(
            f.__closure__ is None,
            msg='f actually has closure cells!',
        )

        g = pickle_depickle(f)

        self.assertTrue(
            g.__closure__ is None,
            msg='g now has closure cells even though f does not',
        )

    def test_empty_cell_preserved(self):
        def f():
            if False:  # pragma: no cover
                cell = None

            def g():
                cell  # NameError, unbound free variable

            return g

        g1 = f()
        with pytest.raises(NameError):
            g1()

        g2 = pickle_depickle(g1)
        with pytest.raises(NameError):
            g2()

    def test_unhashable_closure(self):
        def f():
            s = set((1, 2))  # mutable set is unhashable

            def g():
                return len(s)

            return g

        g = pickle_depickle(f())
        self.assertEqual(g(), 2)

    def test_dynamically_generated_class_that_uses_super(self):

        class Base(object):
            def method(self):
                return 1

        class Derived(Base):
            "Derived Docstring"
            def method(self):
                return super(Derived, self).method() + 1

        self.assertEqual(Derived().method(), 2)

        # Pickle and unpickle the class.
        UnpickledDerived = pickle_depickle(Derived)
        self.assertEqual(UnpickledDerived().method(), 2)

        # We have special logic for handling __doc__ because it's a readonly
        # attribute on PyPy.
        self.assertEqual(UnpickledDerived.__doc__, "Derived Docstring")

        # Pickle and unpickle an instance.
        orig_d = Derived()
        d = pickle_depickle(orig_d)
        self.assertEqual(d.method(), 2)

    def test_cycle_in_classdict_globals(self):

        class C(object):

            def it_works(self):
                return "woohoo!"

        C.C_again = C
        C.instance_of_C = C()

        depickled_C = pickle_depickle(C)
        depickled_instance = pickle_depickle(C())

        # Test instance of depickled class.
        self.assertEqual(depickled_C().it_works(), "woohoo!")
        self.assertEqual(depickled_C.C_again().it_works(), "woohoo!")
        self.assertEqual(depickled_C.instance_of_C.it_works(), "woohoo!")
        self.assertEqual(depickled_instance.it_works(), "woohoo!")

    @pytest.mark.skipif(sys.version_info >= (3, 4)
                        and sys.version_info < (3, 4, 3),
                        reason="subprocess has a bug in 3.4.0 to 3.4.2")
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
        new_class = subprocess_pickle_echo(SomeClass)
        self.assertEqual(new_class(5).some_method(41), 7)

        # pickle the class instances
        self.assertEqual(pickle_depickle(SomeClass(1)).one(), 1)
        self.assertEqual(pickle_depickle(SomeClass(5)).some_method(41), 7)
        new_instance = subprocess_pickle_echo(SomeClass(5))
        self.assertEqual(new_instance.some_method(41), 7)

        # pickle the method instances
        self.assertEqual(pickle_depickle(SomeClass(1).one)(), 1)
        self.assertEqual(pickle_depickle(SomeClass(5).some_method)(41), 7)
        new_method = subprocess_pickle_echo(SomeClass(5).some_method)
        self.assertEqual(new_method(41), 7)

    def test_partial(self):
        partial_obj = functools.partial(min, 1)
        self.assertEqual(pickle_depickle(partial_obj)(4), 1)

    @pytest.mark.skipif(platform.python_implementation() == 'PyPy',
                        reason="Skip numpy and scipy tests on PyPy")
    def test_ufunc(self):
        # test a numpy ufunc (universal function), which is a C-based function
        # that is applied on a numpy array

        if np:
            # simple ufunc: np.add
            self.assertEqual(pickle_depickle(np.add), np.add)
        else:  # skip if numpy is not available
            pass

        if spp:
            # custom ufunc: scipy.special.iv
            self.assertEqual(pickle_depickle(spp.iv), spp.iv)
        else:  # skip if scipy is not available
            pass

    def test_save_unsupported(self):
        sio = StringIO()
        pickler = cloudpickle.CloudPickler(sio, 2)

        with pytest.raises(pickle.PicklingError) as excinfo:
            pickler.save_unsupported("test")

        assert "Cannot pickle objects of type" in str(excinfo.value)

    def test_loads_namespace(self):
        obj = 1, 2, 3, 4
        returned_obj = cloudpickle.loads(cloudpickle.dumps(obj))
        self.assertEqual(obj, returned_obj)

    def test_load_namespace(self):
        obj = 1, 2, 3, 4
        bio = BytesIO()
        cloudpickle.dump(obj, bio)
        bio.seek(0)
        returned_obj = cloudpickle.load(bio)
        self.assertEqual(obj, returned_obj)

    def test_generator(self):

        def some_generator(cnt):
            for i in range(cnt):
                yield i

        gen2 = pickle_depickle(some_generator)

        assert type(gen2(3)) == type(some_generator(3))
        assert list(gen2(3)) == list(range(3))

    def test_classmethod(self):
        class A(object):
            @staticmethod
            def test_sm():
                return "sm"
            @classmethod
            def test_cm(cls):
                return "cm"

        sm = A.__dict__["test_sm"]
        cm = A.__dict__["test_cm"]

        A.test_sm = pickle_depickle(sm)
        A.test_cm = pickle_depickle(cm)

        self.assertEqual(A.test_sm(), "sm")
        self.assertEqual(A.test_cm(), "cm")

    def test_method_descriptors(self):
        f = pickle_depickle(str.upper)
        self.assertEqual(f('abc'), 'ABC')

    def test_instancemethods_without_self(self):
        class F(object):
            def f(self, x):
                return x + 1

        g = pickle_depickle(F.f)
        self.assertEqual(g.__name__, F.f.__name__)
        if sys.version_info[0] < 3:
            self.assertEqual(g.im_class.__name__, F.f.im_class.__name__)
        # self.assertEqual(g(F(), 1), 2)  # still fails

    def test_module(self):
        self.assertEqual(pickle, pickle_depickle(pickle))

    def test_dynamic_module(self):
        mod = imp.new_module('mod')
        code = '''
        x = 1
        def f(y):
            return x + y
        '''
        exec(textwrap.dedent(code), mod.__dict__)
        mod2 = pickle_depickle(mod)
        self.assertEqual(mod.x, mod2.x)
        self.assertEqual(mod.f(5), mod2.f(5))

        # Test dynamic modules when imported back are singletons
        mod1, mod2 = pickle_depickle([mod, mod])
        self.assertEqual(id(mod1), id(mod2))

    def test_find_module(self):
        import pickle  # ensure this test is decoupled from global imports
        _find_module('pickle')

        with pytest.raises(ImportError):
            _find_module('invalid_module')

        with pytest.raises(ImportError):
            valid_module = imp.new_module('valid_module')
            _find_module('valid_module')

    def test_Ellipsis(self):
        self.assertEqual(Ellipsis, pickle_depickle(Ellipsis))

    def test_NotImplemented(self):
        self.assertEqual(NotImplemented, pickle_depickle(NotImplemented))

    @pytest.mark.skipif((3, 0) < sys.version_info < (3, 4),
                        reason="fails due to pickle behavior in Python 3.0-3.3")
    def test_builtin_function_without_module(self):
        on = object.__new__
        on_depickled = pickle_depickle(on)
        self.assertEqual(type(on_depickled(object)), type(object()))

        fi = itertools.chain.from_iterable
        fi_depickled = pickle_depickle(fi)
        self.assertEqual(list(fi([[1, 2], [3, 4]])), [1, 2, 3, 4])

    @pytest.mark.skipif(tornado is None,
                        reason="test needs Tornado installed")
    def test_tornado_coroutine(self):
        # Pickling a locally defined coroutine function
        from tornado import gen, ioloop

        @gen.coroutine
        def f(x, y):
            yield gen.sleep(x)
            raise gen.Return(y + 1)

        @gen.coroutine
        def g(y):
            res = yield f(0.01, y)
            raise gen.Return(res + 1)

        data = cloudpickle.dumps([g, g])
        f = g = None
        g2, g3 = pickle.loads(data)
        self.assertTrue(g2 is g3)
        loop = ioloop.IOLoop.current()
        res = loop.run_sync(functools.partial(g2, 5))
        self.assertEqual(res, 7)

    def test_extended_arg(self):
        # Functions with more than 65535 global vars prefix some global
        # variable references with the EXTENDED_ARG opcode.
        nvars = 65537 + 258
        names = ['g%d' % i for i in range(1, nvars)]
        r = random.Random(42)
        d = dict([(name, r.randrange(100)) for name in names])
        # def f(x):
        #     x = g1, g2, ...
        #     return zlib.crc32(bytes(bytearray(x)))
        code = """
        import zlib

        def f():
            x = {tup}
            return zlib.crc32(bytes(bytearray(x)))
        """.format(tup=', '.join(names))
        exec(textwrap.dedent(code), d, d)
        f = d['f']
        res = f()
        data = cloudpickle.dumps([f, f])
        d = f = None
        f2, f3 = pickle.loads(data)
        self.assertTrue(f2 is f3)
        self.assertEqual(f2(), res)

    def test_submodule(self):
        # Function that refers (by attribute) to a sub-module of a package.

        # Choose any module NOT imported by __init__ of its parent package
        # examples in standard library include:
        # - http.cookies, unittest.mock, curses.textpad, xml.etree.ElementTree

        global xml # imitate performing this import at top of file
        import xml.etree.ElementTree
        def example():
            x = xml.etree.ElementTree.Comment # potential AttributeError

        s = cloudpickle.dumps(example)

        # refresh the environment, i.e., unimport the dependency
        del xml
        for item in list(sys.modules):
            if item.split('.')[0] == 'xml':
                del sys.modules[item]

        # deserialise
        f = pickle.loads(s)
        f() # perform test for error

    def test_submodule_closure(self):
        # Same as test_submodule except the package is not a global
        def scope():
            import xml.etree.ElementTree
            def example():
                x = xml.etree.ElementTree.Comment # potential AttributeError
            return example
        example = scope()

        s = cloudpickle.dumps(example)

        # refresh the environment (unimport dependency)
        for item in list(sys.modules):
            if item.split('.')[0] == 'xml':
                del sys.modules[item]

        f = cloudpickle.loads(s)
        f() # test

    def test_multiprocess(self):
        # running a function pickled by another process (a la dask.distributed)
        def scope():
            import curses.textpad
            def example():
                x = xml.etree.ElementTree.Comment
                x = curses.textpad.Textbox
            return example
        global xml
        import xml.etree.ElementTree
        example = scope()

        s = cloudpickle.dumps(example)

        # choose "subprocess" rather than "multiprocessing" because the latter
        # library uses fork to preserve the parent environment.
        command = ("import pickle, base64; "
                   "pickle.loads(base64.b32decode('" +
                   base64.b32encode(s).decode('ascii') +
                   "'))()")
        assert not subprocess.call([sys.executable, '-c', command])

    def test_import(self):
        # like test_multiprocess except subpackage modules referenced directly
        # (unlike test_submodule)
        global etree
        def scope():
            import curses.textpad as foobar
            def example():
                x = etree.Comment
                x = foobar.Textbox
            return example
        example = scope()
        import xml.etree.ElementTree as etree

        s = cloudpickle.dumps(example)

        command = ("import pickle, base64; "
                   "pickle.loads(base64.b32decode('" +
                   base64.b32encode(s).decode('ascii') +
                   "'))()")
        assert not subprocess.call([sys.executable, '-c', command])

    def test_cell_manipulation(self):
        cell = _make_empty_cell()

        with pytest.raises(ValueError):
            cell.cell_contents

        ob = object()
        cell_set(cell, ob)
        self.assertTrue(
            cell.cell_contents is ob,
            msg='cell contents not set correctly',
        )

    def test_logger(self):
        logger = logging.getLogger('cloudpickle.dummy_test_logger')
        pickled = pickle_depickle(logger)
        self.assertTrue(pickled is logger, (pickled, logger))

        dumped = cloudpickle.dumps(logger)

        code = """if 1:
            import cloudpickle, logging

            logging.basicConfig(level=logging.INFO)
            logger = cloudpickle.loads(%(dumped)r)
            logger.info('hello')
            """ % locals()
        proc = subprocess.Popen([sys.executable, "-c", code],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        self.assertEqual(proc.wait(), 0)
        self.assertEqual(out.strip().decode(),
                         'INFO:cloudpickle.dummy_test_logger:hello')

    def test_abc(self):

        @abc.abstractmethod
        def foo(self):
            raise NotImplementedError('foo')

        # Invoke the metaclass directly rather than using class syntax for
        # python 2/3 compat.
        AbstractClass = abc.ABCMeta('AbstractClass', (object,), {'foo': foo})

        class ConcreteClass(AbstractClass):
            def foo(self):
                return 'it works!'

        depickled_base = pickle_depickle(AbstractClass)
        depickled_class = pickle_depickle(ConcreteClass)
        depickled_instance = pickle_depickle(ConcreteClass())

        self.assertEqual(depickled_class().foo(), 'it works!')
        self.assertEqual(depickled_instance.foo(), 'it works!')

        # assertRaises doesn't return a contextmanager in python 2.6 :(.
        self.failUnlessRaises(TypeError, depickled_base)

        class DepickledBaseSubclass(depickled_base):
            def foo(self):
                return 'it works for realz!'

        self.assertEqual(DepickledBaseSubclass().foo(), 'it works for realz!')

    @pytest.mark.skipif(not HAVE_WEAKSET, reason="WeakSet doesn't exist")
    def test_weakset_identity_preservation(self):
        # Test that weaksets don't lose all their inhabitants if they're
        # pickled in a larger data structure that includes other references to
        # their inhabitants.

        class SomeClass(object):
            def __init__(self, x):
                self.x = x

        obj1, obj2, obj3 = SomeClass(1), SomeClass(2), SomeClass(3)

        things = [weakref.WeakSet([obj1, obj2]), obj1, obj2, obj3]
        result = pickle_depickle(things)

        weakset, depickled1, depickled2, depickled3 = result

        self.assertEqual(depickled1.x, 1)
        self.assertEqual(depickled2.x, 2)
        self.assertEqual(depickled3.x, 3)
        self.assertEqual(len(weakset), 2)

        self.assertEqual(set(weakset), set([depickled1, depickled2]))

    def test_ignoring_whichmodule_exception(self):
        class FakeModule(object):
            def __getattr__(self, name):
                # This throws an exception while looking up within
                # pickle.whichimodule.
                raise Exception()

        class Foo(object):
            __module__ = None

            def foo(self):
                return "it works!"

        def foo():
            return "it works!"

        foo.__module__ = None

        sys.modules["_fake_module"] = FakeModule()
        try:
            # Test whichmodule in save_global.
            self.assertEqual(pickle_depickle(Foo()).foo(), "it works!")

            # Test whichmodule in save_function.
            self.assertEqual(pickle_depickle(foo)(), "it works!")
        finally:
            sys.modules.pop("_fake_module", None)

    def test_dynamic_pytest_module(self):
        # Test case for pull request https://github.com/cloudpipe/cloudpickle/pull/116
        import py

        def f():
            s = py.builtin.set([1])
            return s.pop()

        # some setup is required to allow pytest apimodules to be correctly serializable.
        from cloudpickle import CloudPickler
        from py._apipkg import ApiModule
        CloudPickler.dispatch[ApiModule] = CloudPickler.save_module
        g = cloudpickle.loads(cloudpickle.dumps(f))

        result = g()
        self.assertEqual(1, result)

    def test_function_module_name(self):
        func = lambda x: x
        self.assertEqual(pickle_depickle(func).__module__, func.__module__)

    def test_namedtuple(self):

        MyTuple = collections.namedtuple('MyTuple', ['a', 'b', 'c'])
        t = MyTuple(1, 2, 3)

        depickled_t, depickled_MyTuple = pickle_depickle([t, MyTuple])
        self.assertIsInstance(depickled_t, depickled_MyTuple)

        self.assertEqual((depickled_t.a, depickled_t.b, depickled_t.c), (1, 2, 3))
        self.assertEqual((depickled_t[0], depickled_t[1], depickled_t[2]), (1, 2, 3))

        self.assertEqual(depickled_MyTuple.__name__, 'MyTuple')
        self.assertTrue(issubclass(depickled_MyTuple, tuple))

if __name__ == '__main__':
    unittest.main()
