from __future__ import division

import abc
import collections
import base64
import functools
import io
import itertools
import logging
import math
from operator import itemgetter, attrgetter
import pickle
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import weakref
import os

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
from cloudpickle.cloudpickle import _is_dynamic
from cloudpickle.cloudpickle import _make_empty_cell, cell_set

from .testutils import subprocess_pickle_echo
from .testutils import assert_run_python_script


_TEST_GLOBAL_VARIABLE = "default_value"


class RaiserOnPickle(object):

    def __init__(self, exc):
        self.exc = exc

    def __reduce__(self):
        raise self.exc


def pickle_depickle(obj, protocol=cloudpickle.DEFAULT_PROTOCOL):
    """Helper function to test whether object pickled with cloudpickle can be
    depickled with pickle
    """
    return pickle.loads(cloudpickle.dumps(obj, protocol=protocol))


def _escape(raw_filepath):
    # Ugly hack to embed filepaths in code templates for windows
    return raw_filepath.replace("\\", r"\\\\")


class CloudPickleTest(unittest.TestCase):

    protocol = cloudpickle.DEFAULT_PROTOCOL

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tmp_cloudpickle_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_itemgetter(self):
        d = range(10)
        getter = itemgetter(1)

        getter2 = pickle_depickle(getter, protocol=self.protocol)
        self.assertEqual(getter(d), getter2(d))

        getter = itemgetter(0, 3)
        getter2 = pickle_depickle(getter, protocol=self.protocol)
        self.assertEqual(getter(d), getter2(d))

    def test_attrgetter(self):
        class C(object):
            def __getattr__(self, item):
                return item
        d = C()
        getter = attrgetter("a")
        getter2 = pickle_depickle(getter, protocol=self.protocol)
        self.assertEqual(getter(d), getter2(d))
        getter = attrgetter("a", "b")
        getter2 = pickle_depickle(getter, protocol=self.protocol)
        self.assertEqual(getter(d), getter2(d))

        d.e = C()
        getter = attrgetter("e.a")
        getter2 = pickle_depickle(getter, protocol=self.protocol)
        self.assertEqual(getter(d), getter2(d))
        getter = attrgetter("e.a", "e.b")
        getter2 = pickle_depickle(getter, protocol=self.protocol)
        self.assertEqual(getter(d), getter2(d))

    # Regression test for SPARK-3415
    def test_pickling_file_handles(self):
        out1 = sys.stderr
        out2 = pickle.loads(cloudpickle.dumps(out1, protocol=self.protocol))
        self.assertEqual(out1, out2)

    def test_func_globals(self):
        class Unpicklable(object):
            def __reduce__(self):
                raise Exception("not picklable")

        global exit
        exit = Unpicklable()

        self.assertRaises(Exception, lambda: cloudpickle.dumps(
            exit, protocol=self.protocol))

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
            buffer_clone = pickle_depickle(buffer_obj, protocol=self.protocol)
            self.assertEqual(buffer_clone, str(buffer_obj))
            buffer_obj = buffer("Hello", 2, 3)
            buffer_clone = pickle_depickle(buffer_obj, protocol=self.protocol)
            self.assertEqual(buffer_clone, str(buffer_obj))
        except NameError:  # Python 3 does no longer support buffers
            pass

    def test_memoryview(self):
        buffer_obj = memoryview(b"Hello")
        self.assertEqual(pickle_depickle(buffer_obj, protocol=self.protocol),
                         buffer_obj.tobytes())

    @pytest.mark.skipif(sys.version_info < (3, 4),
                        reason="non-contiguous memoryview not implemented in "
                               "old Python versions")
    def test_sliced_and_non_contiguous_memoryview(self):
        buffer_obj = memoryview(b"Hello!" * 3)[2:15:2]
        self.assertEqual(pickle_depickle(buffer_obj, protocol=self.protocol),
                         buffer_obj.tobytes())

    def test_large_memoryview(self):
        buffer_obj = memoryview(b"Hello!" * int(1e7))
        self.assertEqual(pickle_depickle(buffer_obj, protocol=self.protocol),
                         buffer_obj.tobytes())

    def test_lambda(self):
        self.assertEqual(
                pickle_depickle(lambda: 1, protocol=self.protocol)(), 1)

    def test_nested_lambdas(self):
        a, b = 1, 2
        f1 = lambda x: x + a
        f2 = lambda x: f1(x) // b
        self.assertEqual(pickle_depickle(f2, protocol=self.protocol)(1), 1)

    def test_recursive_closure(self):
        def f1():
            def g():
                return g
            return g

        def f2(base):
            def g(n):
                return base if n <= 1 else n * g(n - 1)
            return g

        g1 = pickle_depickle(f1(), protocol=self.protocol)
        self.assertEqual(g1(), g1)

        g2 = pickle_depickle(f2(2), protocol=self.protocol)
        self.assertEqual(g2(5), 240)

    def test_closure_none_is_preserved(self):
        def f():
            """a function with no closure cells
            """

        self.assertTrue(
            f.__closure__ is None,
            msg='f actually has closure cells!',
        )

        g = pickle_depickle(f, protocol=self.protocol)

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

        g2 = pickle_depickle(g1, protocol=self.protocol)
        with pytest.raises(NameError):
            g2()

    def test_unhashable_closure(self):
        def f():
            s = {1, 2}  # mutable set is unhashable

            def g():
                return len(s)

            return g

        g = pickle_depickle(f(), protocol=self.protocol)
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
        UnpickledDerived = pickle_depickle(Derived, protocol=self.protocol)
        self.assertEqual(UnpickledDerived().method(), 2)

        # We have special logic for handling __doc__ because it's a readonly
        # attribute on PyPy.
        self.assertEqual(UnpickledDerived.__doc__, "Derived Docstring")

        # Pickle and unpickle an instance.
        orig_d = Derived()
        d = pickle_depickle(orig_d, protocol=self.protocol)
        self.assertEqual(d.method(), 2)

    def test_cycle_in_classdict_globals(self):

        class C(object):

            def it_works(self):
                return "woohoo!"

        C.C_again = C
        C.instance_of_C = C()

        depickled_C = pickle_depickle(C, protocol=self.protocol)
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
            # Make sure the __builtins__ are not broken (see #211)
            sum(range(10))
            return (x + y) / LOCAL_CONSTANT

        # pickle the function definition
        self.assertEqual(pickle_depickle(some_function, protocol=self.protocol)(41, 1), 1)
        self.assertEqual(pickle_depickle(some_function, protocol=self.protocol)(81, 3), 2)

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
        clone_class = pickle_depickle(SomeClass, protocol=self.protocol)
        self.assertEqual(clone_class(1).one(), 1)
        self.assertEqual(clone_class(5).some_method(41), 7)
        clone_class = subprocess_pickle_echo(SomeClass, protocol=self.protocol)
        self.assertEqual(clone_class(5).some_method(41), 7)

        # pickle the class instances
        self.assertEqual(pickle_depickle(SomeClass(1)).one(), 1)
        self.assertEqual(pickle_depickle(SomeClass(5)).some_method(41), 7)
        new_instance = subprocess_pickle_echo(SomeClass(5),
                                              protocol=self.protocol)
        self.assertEqual(new_instance.some_method(41), 7)

        # pickle the method instances
        self.assertEqual(pickle_depickle(SomeClass(1).one)(), 1)
        self.assertEqual(pickle_depickle(SomeClass(5).some_method)(41), 7)
        new_method = subprocess_pickle_echo(SomeClass(5).some_method,
                                            protocol=self.protocol)
        self.assertEqual(new_method(41), 7)

    def test_partial(self):
        partial_obj = functools.partial(min, 1)
        partial_clone = pickle_depickle(partial_obj, protocol=self.protocol)
        self.assertEqual(partial_clone(4), 1)

    @pytest.mark.skipif(platform.python_implementation() == 'PyPy',
                        reason="Skip numpy and scipy tests on PyPy")
    def test_ufunc(self):
        # test a numpy ufunc (universal function), which is a C-based function
        # that is applied on a numpy array

        if np:
            # simple ufunc: np.add
            self.assertEqual(pickle_depickle(np.add, protocol=self.protocol),
                             np.add)
        else:  # skip if numpy is not available
            pass

        if spp:
            # custom ufunc: scipy.special.iv
            self.assertEqual(pickle_depickle(spp.iv, protocol=self.protocol),
                             spp.iv)
        else:  # skip if scipy is not available
            pass

    def test_loads_namespace(self):
        obj = 1, 2, 3, 4
        returned_obj = cloudpickle.loads(cloudpickle.dumps(
            obj, protocol=self.protocol))
        self.assertEqual(obj, returned_obj)

    def test_load_namespace(self):
        obj = 1, 2, 3, 4
        bio = io.BytesIO()
        cloudpickle.dump(obj, bio)
        bio.seek(0)
        returned_obj = cloudpickle.load(bio)
        self.assertEqual(obj, returned_obj)

    def test_generator(self):

        def some_generator(cnt):
            for i in range(cnt):
                yield i

        gen2 = pickle_depickle(some_generator, protocol=self.protocol)

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

        A.test_sm = pickle_depickle(sm, protocol=self.protocol)
        A.test_cm = pickle_depickle(cm, protocol=self.protocol)

        self.assertEqual(A.test_sm(), "sm")
        self.assertEqual(A.test_cm(), "cm")

    def test_method_descriptors(self):
        f = pickle_depickle(str.upper)
        self.assertEqual(f('abc'), 'ABC')

    def test_instancemethods_without_self(self):
        class F(object):
            def f(self, x):
                return x + 1

        g = pickle_depickle(F.f, protocol=self.protocol)
        self.assertEqual(g.__name__, F.f.__name__)
        if sys.version_info[0] < 3:
            self.assertEqual(g.im_class.__name__, F.f.im_class.__name__)
        # self.assertEqual(g(F(), 1), 2)  # still fails

    def test_module(self):
        pickle_clone = pickle_depickle(pickle, protocol=self.protocol)
        self.assertEqual(pickle, pickle_clone)

    def test_dynamic_module(self):
        mod = types.ModuleType('mod')
        code = '''
        x = 1
        def f(y):
            return x + y

        class Foo:
            def method(self, x):
                return f(x)
        '''
        exec(textwrap.dedent(code), mod.__dict__)
        mod2 = pickle_depickle(mod, protocol=self.protocol)
        self.assertEqual(mod.x, mod2.x)
        self.assertEqual(mod.f(5), mod2.f(5))
        self.assertEqual(mod.Foo().method(5), mod2.Foo().method(5))

        if platform.python_implementation() != 'PyPy':
            # XXX: this fails with excessive recursion on PyPy.
            mod3 = subprocess_pickle_echo(mod, protocol=self.protocol)
            self.assertEqual(mod.x, mod3.x)
            self.assertEqual(mod.f(5), mod3.f(5))
            self.assertEqual(mod.Foo().method(5), mod3.Foo().method(5))

        # Test dynamic modules when imported back are singletons
        mod1, mod2 = pickle_depickle([mod, mod])
        self.assertEqual(id(mod1), id(mod2))

    def test_module_locals_behavior(self):
        # Makes sure that a local function defined in another module is
        # correctly serialized. This notably checks that the globals are
        # accessible and that there is no issue with the builtins (see #211)

        pickled_func_path = os.path.join(self.tmpdir, 'local_func_g.pkl')

        child_process_script = '''
        import pickle
        import gc
        with open("{pickled_func_path}", 'rb') as f:
            func = pickle.load(f)

        assert func(range(10)) == 45
        '''

        child_process_script = child_process_script.format(
                pickled_func_path=_escape(pickled_func_path))

        try:

            from .testutils import make_local_function

            g = make_local_function()
            with open(pickled_func_path, 'wb') as f:
                cloudpickle.dump(g, f, protocol=self.protocol)

            assert_run_python_script(textwrap.dedent(child_process_script))

        finally:
            os.unlink(pickled_func_path)

    def test_load_dynamic_module_in_grandchild_process(self):
        # Make sure that when loaded, a dynamic module preserves its dynamic
        # property. Otherwise, this will lead to an ImportError if pickled in
        # the child process and reloaded in another one.

        # We create a new dynamic module
        mod = types.ModuleType('mod')
        code = '''
        x = 1
        '''
        exec(textwrap.dedent(code), mod.__dict__)

        # This script will be ran in a separate child process. It will import
        # the pickled dynamic module, and then re-pickle it under a new name.
        # Finally, it will create a child process that will load the re-pickled
        # dynamic module.
        parent_process_module_file = os.path.join(
            self.tmpdir, 'dynamic_module_from_parent_process.pkl')
        child_process_module_file = os.path.join(
            self.tmpdir, 'dynamic_module_from_child_process.pkl')
        child_process_script = '''
            import pickle
            import textwrap

            import cloudpickle
            from testutils import assert_run_python_script


            child_of_child_process_script = {child_of_child_process_script}

            with open('{parent_process_module_file}', 'rb') as f:
                mod = pickle.load(f)

            with open('{child_process_module_file}', 'wb') as f:
                cloudpickle.dump(mod, f, protocol={protocol})

            assert_run_python_script(textwrap.dedent(child_of_child_process_script))
            '''

        # The script ran by the process created by the child process
        child_of_child_process_script = """ '''
                import pickle
                with open('{child_process_module_file}','rb') as fid:
                    mod = pickle.load(fid)
                ''' """

        # Filling the two scripts with the pickled modules filepaths and,
        # for the first child process, the script to be executed by its
        # own child process.
        child_of_child_process_script = child_of_child_process_script.format(
                child_process_module_file=child_process_module_file)

        child_process_script = child_process_script.format(
            parent_process_module_file=_escape(parent_process_module_file),
            child_process_module_file=_escape(child_process_module_file),
            child_of_child_process_script=_escape(child_of_child_process_script),
            protocol=self.protocol)

        try:
            with open(parent_process_module_file, 'wb') as fid:
                cloudpickle.dump(mod, fid, protocol=self.protocol)

            assert_run_python_script(textwrap.dedent(child_process_script))

        finally:
            # Remove temporary created files
            if os.path.exists(parent_process_module_file):
                os.unlink(parent_process_module_file)
            if os.path.exists(child_process_module_file):
                os.unlink(child_process_module_file)

    def test_correct_globals_import(self):
        def nested_function(x):
            return x + 1

        def unwanted_function(x):
            return math.exp(x)

        def my_small_function(x, y):
            return nested_function(x) + y

        b = cloudpickle.dumps(my_small_function, protocol=self.protocol)

        # Make sure that the pickle byte string only includes the definition
        # of my_small_function and its dependency nested_function while
        # extra functions and modules such as unwanted_function and the math
        # module are not included so as to keep the pickle payload as
        # lightweight as possible.

        assert b'my_small_function' in b
        assert b'nested_function' in b

        assert b'unwanted_function' not in b
        assert b'math' not in b

    def test_is_dynamic_module(self):
        import pickle  # decouple this test from global imports
        import os.path
        import distutils
        import distutils.ccompiler

        assert not _is_dynamic(pickle)
        assert not _is_dynamic(os.path)  # fake (aliased) module
        assert not _is_dynamic(distutils)  # package
        assert not _is_dynamic(distutils.ccompiler)  # module in package

        # user-created module without using the import machinery are also
        # dynamic
        dynamic_module = types.ModuleType('dynamic_module')
        assert _is_dynamic(dynamic_module)

    def test_Ellipsis(self):
        self.assertEqual(Ellipsis,
                         pickle_depickle(Ellipsis, protocol=self.protocol))

    def test_NotImplemented(self):
        ExcClone = pickle_depickle(NotImplemented, protocol=self.protocol)
        self.assertEqual(NotImplemented, ExcClone)

    def test_NoneType(self):
        res = pickle_depickle(type(None), protocol=self.protocol)
        self.assertEqual(type(None), res)

    def test_EllipsisType(self):
        res = pickle_depickle(type(Ellipsis), protocol=self.protocol)
        self.assertEqual(type(Ellipsis), res)

    def test_NotImplementedType(self):
        res = pickle_depickle(type(NotImplemented), protocol=self.protocol)
        self.assertEqual(type(NotImplemented), res)

    def test_builtin_function_without_module(self):
        on = object.__new__
        on_depickled = pickle_depickle(on, protocol=self.protocol)
        self.assertEqual(type(on_depickled(object)), type(object()))

        fi = itertools.chain.from_iterable
        fi_depickled = pickle_depickle(fi, protocol=self.protocol)
        self.assertEqual(list(fi_depickled([[1, 2], [3, 4]])), [1, 2, 3, 4])

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

        data = cloudpickle.dumps([g, g], protocol=self.protocol)
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
        d = {name: r.randrange(100) for name in names}
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
        data = cloudpickle.dumps([f, f], protocol=self.protocol)
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

        s = cloudpickle.dumps(example, protocol=self.protocol)

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

        s = cloudpickle.dumps(example, protocol=self.protocol)

        # refresh the environment (unimport dependency)
        for item in list(sys.modules):
            if item.split('.')[0] == 'xml':
                del sys.modules[item]

        f = cloudpickle.loads(s)
        f() # test

    def test_multiprocess(self):
        # running a function pickled by another process (a la dask.distributed)
        def scope():
            def example():
                x = xml.etree.ElementTree.Comment
            return example
        global xml
        import xml.etree.ElementTree
        example = scope()

        s = cloudpickle.dumps(example, protocol=self.protocol)

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
            import xml.etree as foobar
            def example():
                x = etree.Comment
                x = foobar.ElementTree
            return example
        example = scope()
        import xml.etree.ElementTree as etree

        s = cloudpickle.dumps(example, protocol=self.protocol)

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

    def check_logger(self, name):
        logger = logging.getLogger(name)
        pickled = pickle_depickle(logger, protocol=self.protocol)
        self.assertTrue(pickled is logger, (pickled, logger))

        dumped = cloudpickle.dumps(logger)

        code = """if 1:
            import base64, cloudpickle, logging

            logging.basicConfig(level=logging.INFO)
            logger = cloudpickle.loads(base64.b32decode(b'{}'))
            logger.info('hello')
            """.format(base64.b32encode(dumped).decode('ascii'))
        proc = subprocess.Popen([sys.executable, "-c", code],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        self.assertEqual(proc.wait(), 0)
        self.assertEqual(out.strip().decode(),
                         'INFO:{}:hello'.format(logger.name))

    def test_logger(self):
        # logging.RootLogger object
        self.check_logger(None)
        # logging.Logger object
        self.check_logger('cloudpickle.dummy_test_logger')

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

        # This class is local so we can safely register tuple in it to verify
        # the unpickled class also register tuple.
        AbstractClass.register(tuple)

        depickled_base = pickle_depickle(AbstractClass, protocol=self.protocol)
        depickled_class = pickle_depickle(ConcreteClass,
                                          protocol=self.protocol)
        depickled_instance = pickle_depickle(ConcreteClass())

        assert issubclass(tuple, AbstractClass)
        assert issubclass(tuple, depickled_base)

        self.assertEqual(depickled_class().foo(), 'it works!')
        self.assertEqual(depickled_instance.foo(), 'it works!')

        self.assertRaises(TypeError, depickled_base)

        class DepickledBaseSubclass(depickled_base):
            def foo(self):
                return 'it works for realz!'

        self.assertEqual(DepickledBaseSubclass().foo(), 'it works for realz!')

    def test_weakset_identity_preservation(self):
        # Test that weaksets don't lose all their inhabitants if they're
        # pickled in a larger data structure that includes other references to
        # their inhabitants.

        class SomeClass(object):
            def __init__(self, x):
                self.x = x

        obj1, obj2, obj3 = SomeClass(1), SomeClass(2), SomeClass(3)

        things = [weakref.WeakSet([obj1, obj2]), obj1, obj2, obj3]
        result = pickle_depickle(things, protocol=self.protocol)

        weakset, depickled1, depickled2, depickled3 = result

        self.assertEqual(depickled1.x, 1)
        self.assertEqual(depickled2.x, 2)
        self.assertEqual(depickled3.x, 3)
        self.assertEqual(len(weakset), 2)

        self.assertEqual(set(weakset), {depickled1, depickled2})

    def test_faulty_module(self):
        for module_name in ['_faulty_module', '_missing_module', None]:
            class FaultyModule(object):
                def __getattr__(self, name):
                    # This throws an exception while looking up within
                    # pickle.whichmodule or getattr(module, name, None)
                    raise Exception()

            class Foo(object):
                __module__ = module_name

                def foo(self):
                    return "it works!"

            def foo():
                return "it works!"

            foo.__module__ = module_name

            sys.modules["_faulty_module"] = FaultyModule()
            try:
                # Test whichmodule in save_global.
                self.assertEqual(pickle_depickle(Foo()).foo(), "it works!")

                # Test whichmodule in save_function.
                cloned = pickle_depickle(foo, protocol=self.protocol)
                self.assertEqual(cloned(), "it works!")
            finally:
                sys.modules.pop("_faulty_module", None)

    def test_dynamic_pytest_module(self):
        # Test case for pull request https://github.com/cloudpipe/cloudpickle/pull/116
        import py

        def f():
            s = py.builtin.set([1])
            return s.pop()

        # some setup is required to allow pytest apimodules to be correctly
        # serializable.
        from cloudpickle import CloudPickler
        CloudPickler.dispatch[type(py.builtin)] = CloudPickler.save_module
        g = cloudpickle.loads(cloudpickle.dumps(f, protocol=self.protocol))

        result = g()
        self.assertEqual(1, result)

    def test_function_module_name(self):
        func = lambda x: x
        cloned = pickle_depickle(func, protocol=self.protocol)
        self.assertEqual(cloned.__module__, func.__module__)

    def test_function_qualname(self):
        def func(x):
            return x
        # Default __qualname__ attribute (Python 3 only)
        if hasattr(func, '__qualname__'):
            cloned = pickle_depickle(func, protocol=self.protocol)
            self.assertEqual(cloned.__qualname__, func.__qualname__)

        # Mutated __qualname__ attribute
        func.__qualname__ = '<modifiedlambda>'
        cloned = pickle_depickle(func, protocol=self.protocol)
        self.assertEqual(cloned.__qualname__, func.__qualname__)

    def test_namedtuple(self):

        MyTuple = collections.namedtuple('MyTuple', ['a', 'b', 'c'])
        t = MyTuple(1, 2, 3)

        depickled_t, depickled_MyTuple = pickle_depickle(
            [t, MyTuple], protocol=self.protocol)
        self.assertTrue(isinstance(depickled_t, depickled_MyTuple))

        self.assertEqual((depickled_t.a, depickled_t.b, depickled_t.c),
                         (1, 2, 3))
        self.assertEqual((depickled_t[0], depickled_t[1], depickled_t[2]),
                         (1, 2, 3))

        self.assertEqual(depickled_MyTuple.__name__, 'MyTuple')
        self.assertTrue(issubclass(depickled_MyTuple, tuple))

    def test_builtin_type__new__(self):
        # Functions occasionally take the __new__ of these types as default
        # parameters for factories.  For example, on Python 3.3,
        # `tuple.__new__` is a default value for some methods of namedtuple.
        for t in list, tuple, set, frozenset, dict, object:
            cloned = pickle_depickle(t.__new__, protocol=self.protocol)
            self.assertTrue(cloned is t.__new__)

    def test_interactively_defined_function(self):
        # Check that callables defined in the __main__ module of a Python
        # script (or jupyter kernel) can be pickled / unpickled / executed.
        code = """\
        from testutils import subprocess_pickle_echo

        CONSTANT = 42

        class Foo(object):

            def method(self, x):
                return x

        foo = Foo()

        def f0(x):
            return x ** 2

        def f1():
            return Foo

        def f2(x):
            return Foo().method(x)

        def f3():
            return Foo().method(CONSTANT)

        def f4(x):
            return foo.method(x)

        def f5(x):
            # Recursive call to a dynamically defined function.
            if x <= 0:
                return f4(x)
            return f5(x - 1) + 1

        cloned = subprocess_pickle_echo(lambda x: x**2, protocol={protocol})
        assert cloned(3) == 9

        cloned = subprocess_pickle_echo(f0, protocol={protocol})
        assert cloned(3) == 9

        cloned = subprocess_pickle_echo(Foo, protocol={protocol})
        assert cloned().method(2) == Foo().method(2)

        cloned = subprocess_pickle_echo(Foo(), protocol={protocol})
        assert cloned.method(2) == Foo().method(2)

        cloned = subprocess_pickle_echo(f1, protocol={protocol})
        assert cloned()().method('a') == f1()().method('a')

        cloned = subprocess_pickle_echo(f2, protocol={protocol})
        assert cloned(2) == f2(2)

        cloned = subprocess_pickle_echo(f3, protocol={protocol})
        assert cloned() == f3()

        cloned = subprocess_pickle_echo(f4, protocol={protocol})
        assert cloned(2) == f4(2)

        cloned = subprocess_pickle_echo(f5, protocol={protocol})
        assert cloned(7) == f5(7) == 7
        """.format(protocol=self.protocol)
        assert_run_python_script(textwrap.dedent(code))

    def test_interactively_defined_global_variable(self):
        # Check that callables defined in the __main__ module of a Python
        # script (or jupyter kernel) correctly retrieve global variables.
        code_template = """\
        from testutils import subprocess_pickle_echo
        from cloudpickle import dumps, loads

        def local_clone(obj, protocol=None):
            return loads(dumps(obj, protocol=protocol))

        VARIABLE = "default_value"

        def f0():
            global VARIABLE
            VARIABLE = "changed_by_f0"

        def f1():
            return VARIABLE

        assert f0.__globals__ is f1.__globals__

        # pickle f0 and f1 inside the same pickle_string
        cloned_f0, cloned_f1 = {clone_func}([f0, f1], protocol={protocol})

        # cloned_f0 and cloned_f1 now share a global namespace that is isolated
        # from any previously existing namespace
        assert cloned_f0.__globals__ is cloned_f1.__globals__
        assert cloned_f0.__globals__ is not f0.__globals__

        # pickle f1 another time, but in a new pickle string
        pickled_f1 = dumps(f1, protocol={protocol})

        # Change the value of the global variable in f0's new global namespace
        cloned_f0()

        # thanks to cloudpickle isolation, depickling and calling f0 and f1
        # should not affect the globals of already existing modules
        assert VARIABLE == "default_value", VARIABLE

        # Ensure that cloned_f1 and cloned_f0 share the same globals, as f1 and
        # f0 shared the same globals at pickling time, and cloned_f1 was
        # depickled from the same pickle string as cloned_f0
        shared_global_var = cloned_f1()
        assert shared_global_var == "changed_by_f0", shared_global_var

        # f1 is unpickled another time, but because it comes from another
        # pickle string than pickled_f1 and pickled_f0, it will not share the
        # same globals as the latter two.
        new_cloned_f1 = loads(pickled_f1)
        assert new_cloned_f1.__globals__ is not cloned_f1.__globals__
        assert new_cloned_f1.__globals__ is not f1.__globals__

        # get the value of new_cloned_f1's VARIABLE
        new_global_var = new_cloned_f1()
        assert new_global_var == "default_value", new_global_var
        """
        for clone_func in ['local_clone', 'subprocess_pickle_echo']:
            code = code_template.format(protocol=self.protocol,
                                        clone_func=clone_func)
            assert_run_python_script(textwrap.dedent(code))

    def test_closure_interacting_with_a_global_variable(self):
        global _TEST_GLOBAL_VARIABLE
        assert _TEST_GLOBAL_VARIABLE == "default_value"
        orig_value = _TEST_GLOBAL_VARIABLE
        try:
            def f0():
                global _TEST_GLOBAL_VARIABLE
                _TEST_GLOBAL_VARIABLE = "changed_by_f0"

            def f1():
                return _TEST_GLOBAL_VARIABLE

            # pickle f0 and f1 inside the same pickle_string
            cloned_f0, cloned_f1 = pickle_depickle([f0, f1],
                                                   protocol=self.protocol)

            # cloned_f0 and cloned_f1 now share a global namespace that is
            # isolated from any previously existing namespace
            assert cloned_f0.__globals__ is cloned_f1.__globals__
            assert cloned_f0.__globals__ is not f0.__globals__

            # pickle f1 another time, but in a new pickle string
            pickled_f1 = cloudpickle.dumps(f1, protocol=self.protocol)

            # Change the global variable's value in f0's new global namespace
            cloned_f0()

            # depickling f0 and f1 should not affect the globals of already
            # existing modules
            assert _TEST_GLOBAL_VARIABLE == "default_value"

            # Ensure that cloned_f1 and cloned_f0 share the same globals, as f1
            # and f0 shared the same globals at pickling time, and cloned_f1
            # was depickled from the same pickle string as cloned_f0
            shared_global_var = cloned_f1()
            assert shared_global_var == "changed_by_f0", shared_global_var

            # f1 is unpickled another time, but because it comes from another
            # pickle string than pickled_f1 and pickled_f0, it will not share
            # the same globals as the latter two.
            new_cloned_f1 = pickle.loads(pickled_f1)
            assert new_cloned_f1.__globals__ is not cloned_f1.__globals__
            assert new_cloned_f1.__globals__ is not f1.__globals__

            # get the value of new_cloned_f1's VARIABLE
            new_global_var = new_cloned_f1()
            assert new_global_var == "default_value", new_global_var
        finally:
            _TEST_GLOBAL_VARIABLE = orig_value

    def test_interactive_remote_function_calls(self):
        code = """if __name__ == "__main__":
        from testutils import subprocess_worker

        def interactive_function(x):
            return x + 1

        with subprocess_worker(protocol={protocol}) as w:

            assert w.run(interactive_function, 41) == 42

            # Define a new function that will call an updated version of
            # the previously called function:

            def wrapper_func(x):
                return interactive_function(x)

            def interactive_function(x):
                return x - 1

            # The change in the definition of interactive_function in the main
            # module of the main process should be reflected transparently
            # in the worker process: the worker process does not recall the
            # previous definition of `interactive_function`:

            assert w.run(wrapper_func, 41) == 40
        """.format(protocol=self.protocol)
        assert_run_python_script(code)

    def test_interactive_remote_function_calls_no_side_effect(self):
        code = """if __name__ == "__main__":
        from testutils import subprocess_worker
        import sys

        with subprocess_worker(protocol={protocol}) as w:

            GLOBAL_VARIABLE = 0

            class CustomClass(object):

                def mutate_globals(self):
                    global GLOBAL_VARIABLE
                    GLOBAL_VARIABLE += 1
                    return GLOBAL_VARIABLE

            custom_object = CustomClass()
            assert w.run(custom_object.mutate_globals) == 1

            # The caller global variable is unchanged in the main process.

            assert GLOBAL_VARIABLE == 0

            # Calling the same function again starts again from zero. The
            # worker process is stateless: it has no memory of the past call:

            assert w.run(custom_object.mutate_globals) == 1

            # The symbols defined in the main process __main__ module are
            # not set in the worker process main module to leave the worker
            # as stateless as possible:

            def is_in_main(name):
                return hasattr(sys.modules["__main__"], name)

            assert is_in_main("CustomClass")
            assert not w.run(is_in_main, "CustomClass")

            assert is_in_main("GLOBAL_VARIABLE")
            assert not w.run(is_in_main, "GLOBAL_VARIABLE")

        """.format(protocol=self.protocol)
        assert_run_python_script(code)

    @pytest.mark.skipif(platform.python_implementation() == 'PyPy',
                        reason="Skip PyPy because memory grows too much")
    def test_interactive_remote_function_calls_no_memory_leak(self):
        code = """if __name__ == "__main__":
        from testutils import subprocess_worker
        import struct

        with subprocess_worker(protocol={protocol}) as w:

            reference_size = w.memsize()
            assert reference_size > 0


            def make_big_closure(i):
                # Generate a byte string of size 1MB
                itemsize = len(struct.pack("l", 1))
                data = struct.pack("l", i) * (int(1e6) // itemsize)
                def process_data():
                    return len(data)
                return process_data

            for i in range(100):
                func = make_big_closure(i)
                result = w.run(func)
                assert result == int(1e6), result

            import gc
            w.run(gc.collect)

            # By this time the worker process has processed worth of 100MB of
            # data passed in the closures its memory size should now have
            # grown by more than a few MB.
            growth = w.memsize() - reference_size
            assert growth < 1e7, growth

        """.format(protocol=self.protocol)
        assert_run_python_script(code)

    @pytest.mark.skipif(sys.version_info >= (3, 0),
                        reason="hardcoded pickle bytes for 2.7")
    def test_function_pickle_compat_0_4_0(self):
        # The result of `cloudpickle.dumps(lambda x: x)` in cloudpickle 0.4.0,
        # Python 2.7
        pickled = (b'\x80\x02ccloudpickle.cloudpickle\n_fill_function\nq\x00(c'
            b'cloudpickle.cloudpickle\n_make_skel_func\nq\x01ccloudpickle.clou'
            b'dpickle\n_builtin_type\nq\x02U\x08CodeTypeq\x03\x85q\x04Rq\x05(K'
            b'\x01K\x01K\x01KCU\x04|\x00\x00Sq\x06N\x85q\x07)U\x01xq\x08\x85q'
            b'\tU\x07<stdin>q\nU\x08<lambda>q\x0bK\x01U\x00q\x0c))tq\rRq\x0eJ'
            b'\xff\xff\xff\xff}q\x0f\x87q\x10Rq\x11}q\x12N}q\x13NtR.')
        self.assertEqual(42, cloudpickle.loads(pickled)(42))

    @pytest.mark.skipif(sys.version_info >= (3, 0),
                        reason="hardcoded pickle bytes for 2.7")
    def test_function_pickle_compat_0_4_1(self):
        # The result of `cloudpickle.dumps(lambda x: x)` in cloudpickle 0.4.1,
        # Python 2.7
        pickled = (b'\x80\x02ccloudpickle.cloudpickle\n_fill_function\nq\x00(c'
            b'cloudpickle.cloudpickle\n_make_skel_func\nq\x01ccloudpickle.clou'
            b'dpickle\n_builtin_type\nq\x02U\x08CodeTypeq\x03\x85q\x04Rq\x05(K'
            b'\x01K\x01K\x01KCU\x04|\x00\x00Sq\x06N\x85q\x07)U\x01xq\x08\x85q'
            b'\tU\x07<stdin>q\nU\x08<lambda>q\x0bK\x01U\x00q\x0c))tq\rRq\x0eJ'
            b'\xff\xff\xff\xff}q\x0f\x87q\x10Rq\x11}q\x12N}q\x13U\x08__main__q'
            b'\x14NtR.')
        self.assertEqual(42, cloudpickle.loads(pickled)(42))

    def test_pickle_reraise(self):
        for exc_type in [Exception, ValueError, TypeError, RuntimeError]:
            obj = RaiserOnPickle(exc_type("foo"))
            with pytest.raises((exc_type, pickle.PicklingError)):
                cloudpickle.dumps(obj, protocol=self.protocol)

    def test_unhashable_function(self):
        d = {'a': 1}
        depickled_method = pickle_depickle(d.get, protocol=self.protocol)
        self.assertEqual(depickled_method('a'), 1)
        self.assertEqual(depickled_method('b'), None)

    def test_itertools_count(self):
        counter = itertools.count(1, step=2)

        # advance the counter a bit
        next(counter)
        next(counter)

        new_counter = pickle_depickle(counter, protocol=self.protocol)

        self.assertTrue(counter is not new_counter)

        for _ in range(10):
            self.assertEqual(next(counter), next(new_counter))

    def test_wraps_preserves_function_name(self):
        from functools import wraps

        def f():
            pass

        @wraps(f)
        def g():
            f()

        f2 = pickle_depickle(g, protocol=self.protocol)

        self.assertEqual(f2.__name__, f.__name__)

    def test_wraps_preserves_function_doc(self):
        from functools import wraps

        def f():
            """42"""
            pass

        @wraps(f)
        def g():
            f()

        f2 = pickle_depickle(g, protocol=self.protocol)

        self.assertEqual(f2.__doc__, f.__doc__)

    @unittest.skipIf(sys.version_info < (3, 7),
                     """This syntax won't work on py2 and pickling annotations
                     isn't supported for py36 and below.""")
    def test_wraps_preserves_function_annotations(self):
        from functools import wraps

        def f(x):
            pass

        f.__annotations__ = {'x': 1, 'return': float}

        @wraps(f)
        def g(x):
            f(x)

        f2 = pickle_depickle(g, protocol=self.protocol)

        self.assertEqual(f2.__annotations__, f.__annotations__)

    def test_instance_with_slots(self):
        for slots in [["registered_attribute"], "registered_attribute"]:
            class ClassWithSlots(object):
                __slots__ = slots

                def __init__(self):
                    self.registered_attribute = 42

            initial_obj = ClassWithSlots()
            depickled_obj = pickle_depickle(
                initial_obj, protocol=self.protocol)

            for obj in [initial_obj, depickled_obj]:
                self.assertEqual(obj.registered_attribute, 42)
                with pytest.raises(AttributeError):
                    obj.non_registered_attribute = 1

    @unittest.skipIf(not hasattr(types, "MappingProxyType"),
                     "Old versions of Python do not have this type.")
    def test_mappingproxy(self):
        mp = types.MappingProxyType({"some_key": "some value"})
        assert mp == pickle_depickle(mp, protocol=self.protocol)

    def test_dataclass(self):
        dataclasses = pytest.importorskip("dataclasses")

        DataClass = dataclasses.make_dataclass('DataClass', [('x', int)])
        data = DataClass(x=42)

        pickle_depickle(DataClass, protocol=self.protocol)
        assert data.x == pickle_depickle(data, protocol=self.protocol).x == 42

    def test_relative_import_inside_function(self):
        # Make sure relative imports inside round-tripped functions is not
        # broken.This was a bug in cloudpickle versions <= 0.5.3 and was
        # re-introduced in 0.8.0.

        # Both functions living inside modules and packages are tested.
        def f():
            # module_function belongs to mypkg.mod1, which is a module
            from .mypkg import module_function
            return module_function()

        def g():
            # package_function belongs to mypkg, which is a package
            from .mypkg import package_function
            return package_function()

        for func, source in zip([f, g], ["module", "package"]):
            # Make sure relative imports are initially working
            assert func() == "hello from a {}!".format(source)

            # Make sure relative imports still work after round-tripping
            cloned_func = pickle_depickle(func, protocol=self.protocol)
            assert cloned_func() == "hello from a {}!".format(source)


class Protocol2CloudPickleTest(CloudPickleTest):

    protocol = 2


if __name__ == '__main__':
    unittest.main()
