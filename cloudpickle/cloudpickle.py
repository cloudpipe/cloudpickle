"""
This class is defined to override standard pickle functionality

The goals of it follow:
-Serialize lambdas and nested functions to compiled byte code
-Deal with main module correctly
-Deal with other non-serializable objects

It does not include an unpickler, as standard python unpickling suffices.

This module was extracted from the `cloud` package, developed by `PiCloud, Inc.
<https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.

Copyright (c) 2012, Regents of the University of California.
Copyright (c) 2009 `PiCloud, Inc. <https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the University of California, Berkeley nor the
      names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import print_function

import codecs
import ctypes
import dis
from functools import partial
import imp
import io
import itertools
import logging
import opcode
import operator
import platform
import pickle
from struct import pack
import sys
import traceback
import types
import weakref


RUNNING_CPYTHON = platform.python_implementation() == 'CPython'

# cloudpickle is meant for inter process communication: we expect all
# communicating processes to run the same Python version hence we favor
# communication speed over compatibility:
DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL

PY_MAJOR_MINOR = sys.version_info[:2]

if PY_MAJOR_MINOR[0] < 3:
    from pickle import Pickler
    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO
    PY3 = False
else:
    types.ClassType = type
    from pickle import _Pickler as Pickler
    from io import BytesIO as StringIO
    PY3 = True


def _make_cell_set_template_code():
    """Get the Python compiler to emit LOAD_FAST(arg); STORE_DEREF

    Notes
    -----
    In Python 3, we could use an easier function:

    .. code-block:: python

       def f():
           cell = None

           def _stub(value):
               nonlocal cell
               cell = value

           return _stub

        _cell_set_template_code = f()

    This function is _only_ a LOAD_FAST(arg); STORE_DEREF, but that is
    invalid syntax on Python 2. If we use this function we also don't need
    to do the weird freevars/cellvars swap below
    """
    def inner(value):
        lambda: cell  # make ``cell`` a closure so that we get a STORE_DEREF
        cell = value

    co = inner.__code__

    # NOTE: we are marking the cell variable as a free variable intentionally
    # so that we simulate an inner function instead of the outer function. This
    # is what gives us the ``nonlocal`` behavior in a Python 2 compatible way.
    if not PY3:
        return types.CodeType(
            co.co_argcount,
            co.co_nlocals,
            co.co_stacksize,
            co.co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_cellvars,  # this is the trickery
            (),
        )
    else:
        return types.CodeType(
            co.co_argcount,
            co.co_kwonlyargcount,
            co.co_nlocals,
            co.co_stacksize,
            co.co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_cellvars,  # this is the trickery
            (),
        )


_cell_set_template_code = _make_cell_set_template_code()


def cell_set(cell, value):
    """Set the value of a closure cell.
    """
    return types.FunctionType(
        _cell_set_template_code,
        {},
        '_cell_set_inner',
        (),
        (cell,),
    )(value)


#relevant opcodes
STORE_GLOBAL = opcode.opmap['STORE_GLOBAL']
DELETE_GLOBAL = opcode.opmap['DELETE_GLOBAL']
LOAD_GLOBAL = opcode.opmap['LOAD_GLOBAL']
GLOBAL_OPS = (STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL)
HAVE_ARGUMENT = dis.HAVE_ARGUMENT
EXTENDED_ARG = dis.EXTENDED_ARG


def islambda(func):
    return getattr(func,'__name__') == '<lambda>'


_BUILTIN_TYPE_NAMES = {}
for k, v in types.__dict__.items():
    if type(v) is type:
        _BUILTIN_TYPE_NAMES[v] = k


def _builtin_type(name):
    return getattr(types, name)


def _make__new__factory(type_):
    def _factory():
        return type_.__new__
    return _factory


# NOTE: These need to be module globals so that they're pickleable as globals.
_get_dict_new = _make__new__factory(dict)
_get_frozenset_new = _make__new__factory(frozenset)
_get_list_new = _make__new__factory(list)
_get_set_new = _make__new__factory(set)
_get_tuple_new = _make__new__factory(tuple)
_get_object_new = _make__new__factory(object)

# Pre-defined set of builtin_function_or_method instances that can be
# serialized.
_BUILTIN_TYPE_CONSTRUCTORS = {
    dict.__new__: _get_dict_new,
    frozenset.__new__: _get_frozenset_new,
    set.__new__: _get_set_new,
    list.__new__: _get_list_new,
    tuple.__new__: _get_tuple_new,
    object.__new__: _get_object_new,
}


if sys.version_info < (3, 4):
    def _walk_global_ops(code):
        """
        Yield (opcode, argument number) tuples for all
        global-referencing instructions in *code*.
        """
        code = getattr(code, 'co_code', b'')
        if not PY3:
            code = map(ord, code)

        n = len(code)
        i = 0
        extended_arg = 0
        while i < n:
            op = code[i]
            i += 1
            if op >= HAVE_ARGUMENT:
                oparg = code[i] + code[i + 1] * 256 + extended_arg
                extended_arg = 0
                i += 2
                if op == EXTENDED_ARG:
                    extended_arg = oparg * 65536
                if op in GLOBAL_OPS:
                    yield op, oparg

else:
    def _walk_global_ops(code):
        """
        Yield (opcode, argument number) tuples for all
        global-referencing instructions in *code*.
        """
        for instr in dis.get_instructions(code):
            op = instr.opcode
            if op in GLOBAL_OPS:
                yield op, instr.arg


class _Py2StrStruct(ctypes.Structure):
    """Map the memory layout of a CPython 2 str to access its interned state"""
    _fields_ = [
        ("ob_refcnt", ctypes.c_long),
        ("ob_type", ctypes.c_void_p),
        ("ob_size", ctypes.c_ulong),
        ('ob_shash', ctypes.c_long),
        ('ob_sstate', ctypes.c_int),
        ('ob_sval', ctypes.c_char),
    ]

    def is_interned(self):
        return self.ob_sstate != 0


def _is_safe_to_mutate(data_holder):
    """Helper function, see _memoryview_from_bytes for details."""
    if not RUNNING_CPYTHON:
        # sys.getrefcount and ob_sstate are not always available on other
        # platforms (e.g. PyPy): better be conservative.
        return False

    if len(data_holder[0]) <= 1:
        # Under CPython 3, single byte bytes can be mapped to singletons which
        # are not safe to mutate.
        return False

    # If the following call to sys.getrefcount returns 2 it means that
    # there is one reference from data_holder and one from the temporary
    # arguments datastructure of the sys.getrefcount call. It is therefore
    # safe to reuse the memory buffer of the bytes object as writeable
    # buffer to back the memoryview: this buffer is no longer reachable
    # from anywhere else.
    safe_to_mutate = sys.getrefcount(data_holder[0]) <= 2
    data = data_holder[0]
    if safe_to_mutate and isinstance(data, str):
        # Python 2 str instances are bytes that can be interned, make sure
        # that this is not the case.
        safe_to_mutate = not _Py2StrStruct.from_address(id(data)).is_interned()
    return safe_to_mutate


def _memoryview_from_bytes(data_holder, format, readonly, shape, **kwargs):
    """Build a memoryview from raw bytes read from the pickle stream

    This function tries its best not to allocate any extra memory when not
    necessary.

    The actual data is provided in a data_holder which is expected to be a
    single item list holding the actual bytes object. This makes it easier
    to use sys.getrefcount to detect whether or not the bytes are referenced
    elsewhere or not.

    If readonly is False and the backing bytes in data_holder are referenced
    from other objects, then the bytes contents are copied into a newly
    allocated mutable buffer. Otherwise the memory allocated for the bytes
    object is directly reused by the new memoryview.

    Python 2 memoryviews doe not support being cast so that the
    format and shape parameters are ignored.

    Reference counting cannot be used in PyPy to detect if it is safe to mutate
    the bytes object. If readonly is False, a new mutable buffer is always
    allocated.

    **kwargs are left ignored. Those are used to add flexibility to implement
    backward compatible changes in future versions of cloudpickle.
    """
    safe_to_mutate = _is_safe_to_mutate(data_holder)
    data = data_holder[0]
    del data_holder[:]
    if readonly:
        # A memoryview of a bytes object is readonly by default.
        view = memoryview(data)
    else:
        # Python 3.4 implementation of memoryviews backed by ctypes buffers
        # has a bug: # https://bugs.python.org/issue19803
        if safe_to_mutate and PY_MAJOR_MINOR != (3, 4):
            buffer = ctypes.cast(data, ctypes.POINTER(ctypes.c_char))
            addr = ctypes.addressof(buffer.contents)
            array = (ctypes.c_char * len(data)).from_address(addr)
            # Store a reference to the backing bytes object to make GC work
            # correctly. We hide the object itself in a closure to prevent
            # unsuspecting users to access it.

            def _hidden_buffer_ref():
                raise TypeError("Access to mutable buffer with id %d is unsafe"
                                % id(data))
            array._hidden_buffer_ref = _hidden_buffer_ref
            view = memoryview(array)
        else:
            # This buffer can be referenced by external objects: for instance
            # empty and single bytes objects are interned under Python 3.
            # Longer bytes objects in Python (from the str type) can also be
            # interned. Force a copy into a new mutable buffer to avoid a
            # violation of the imnmutability of bytes.
            view = memoryview(bytearray(data))

    if PY_MAJOR_MINOR >= (3, 4):
        return view.cast('B').cast(format, shape)
    else:
        # Python 2.7 does not support casting.
        return view


class _Framer:
    """Backport of Python 3.7 framer"""

    _FRAME_SIZE_TARGET = 64 * 1024

    def __init__(self, file_write):
        self.file_write = file_write
        self.current_frame = None

    def start_framing(self):
        self.current_frame = io.BytesIO()

    def end_framing(self):
        if self.current_frame and self.current_frame.tell() > 0:
            self.commit_frame(force=True)
            self.current_frame = None

    def commit_frame(self, force=False):
        if self.current_frame:
            f = self.current_frame
            if f.tell() >= self._FRAME_SIZE_TARGET or force:
                data = f.getbuffer()
                write = self.file_write
                # Issue a single call to the write nethod of the underlying
                # file object for the frame opcode with the size of the
                # frame. The concatenation is expected to be less expensive
                # than issuing an additional call to write.
                write(pickle.FRAME + pack("<Q", len(data)))

                # Issue a separate call to write to append the frame
                # contents without concatenation to the above to avoid a
                # memory copy.
                write(data)

                # Start the new frame with a new io.BytesIO instance so that
                # the file object can have delayed access to the previous frame
                # contents via an unreleased memoryview of the previous
                # io.BytesIO instance.
                self.current_frame = io.BytesIO()

    def write(self, data):
        if self.current_frame:
            return self.current_frame.write(data)
        else:
            return self.file_write(data)

    def write_large_bytes(self, header, payload):
        write = self.file_write
        if self.current_frame:
            # Terminate the current frame and flush it to the file.
            self.commit_frame(force=True)

        # Perform direct write of the header and payload of the large binary
        # object. Be careful not to concatenate the header and the payload
        # prior to calling 'write' as we do not want to allocate a large
        # temporary bytes object.
        # We intentionally do not insert a protocol 4 frame opcode to make
        # it possible to optimize file.read calls in the loader.
        write(header)
        write(payload)


class CloudPickler(Pickler):

    dispatch = Pickler.dispatch.copy()

    def __init__(self, file, protocol=None):
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        Pickler.__init__(self, file, protocol=protocol)
        # set of modules to unpickle
        self.modules = set()
        # map ids to dictionary. used to ensure that functions can share global env
        self.globals_ref = {}

        # Override the framer backport support for no-copy write of large
        # binary data
        self.framer = _Framer(file.write)
        self.write = self.framer.write
        self._write_large_bytes = self.framer.write_large_bytes

    def dump(self, obj):
        self.inject_addons()
        try:
            return Pickler.dump(self, obj)
        except RuntimeError as e:
            if 'recursion' in e.args[0]:
                msg = """Could not pickle object as excessively deep recursion required."""
                raise pickle.PicklingError(msg)

    if PY3:
        def save_bytes(self, obj, skip_memoize=False):
            if self.proto < 3:
                if not obj:  # bytes object is empty
                    self.save_reduce(bytes, (), obj=obj)
                else:
                    self.save_reduce(codecs.encode,
                                     (str(obj, 'latin1'), 'latin1'), obj=obj)
                return
            n = getattr(obj, 'nbytes', len(obj))
            if n <= 0xff:
                self.write(pickle.SHORT_BINBYTES + pack("<B", n) + obj)
            elif n > 0xffffffff and self.proto >= 4:
                self._write_large_bytes(pickle.BINBYTES8 + pack("<Q", n), obj)
            elif n >= self.framer._FRAME_SIZE_TARGET:
                self._write_large_bytes(pickle.BINBYTES + pack("<I", n), obj)
            else:
                self.write(pickle.BINBYTES + pack("<I", n) + obj)
            if not skip_memoize:
                self.memoize(obj)
        dispatch[bytes] = save_bytes

    def save_memoryview(self, obj):
        # Python 2 views doe not have nbytes
        n = getattr(obj, 'nbytes', len(obj))
        self.save(_memoryview_from_bytes)
        self.write(pickle.MARK)
        # wrap bytes buffer in a singleton list that is expected to hold
        # the only reference to the bytes obj when unpickling.
        self.write(pickle.MARK)
        c_contiguous = getattr(obj, 'c_contiguous', False)
        if n <= 0xff or self.proto < 3 or not c_contiguous:
            # Make a contiguous copy as a bytes object prior to serialization.
            # TODO: find a way to disable memoization in this case. Memoization
            # will prevent _memoryview_from_bytes to ensure safe mutability
            # via reference counting.
            self.save(obj.tobytes())
        else:
            # Large contiguous views can benefit from nocopy semantics with
            # recent versions of the pickle protocol.
            self.save_bytes(obj, skip_memoize=True)
        self.write(pickle.LIST)  # data_holder
        self.save(obj.format)
        self.save(obj.readonly)
        self.save(obj.shape)
        self.write(pickle.TUPLE)
        self.write(pickle.REDUCE)
        self.memoize(obj)
    dispatch[memoryview] = save_memoryview

    if not PY3:
        def save_buffer(self, obj):
            self.save(str(obj))
        dispatch[buffer] = save_buffer

    def save_unsupported(self, obj):
        raise pickle.PicklingError("Cannot pickle objects of type %s" % type(obj))
    dispatch[types.GeneratorType] = save_unsupported

    # itertools objects do not pickle!
    for v in itertools.__dict__.values():
        if type(v) is type:
            dispatch[v] = save_unsupported

    def save_module(self, obj):
        """
        Save a module as an import
        """
        mod_name = obj.__name__
        # If module is successfully found then it is not a dynamically created module
        if hasattr(obj, '__file__'):
            is_dynamic = False
        else:
            try:
                _find_module(mod_name)
                is_dynamic = False
            except ImportError:
                is_dynamic = True

        self.modules.add(obj)
        if is_dynamic:
            self.save_reduce(dynamic_subimport, (obj.__name__, vars(obj)), obj=obj)
        else:
            self.save_reduce(subimport, (obj.__name__,), obj=obj)
    dispatch[types.ModuleType] = save_module

    def save_codeobject(self, obj):
        """
        Save a code object
        """
        if PY3:
            args = (
                obj.co_argcount, obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
                obj.co_flags, obj.co_code, obj.co_consts, obj.co_names, obj.co_varnames,
                obj.co_filename, obj.co_name, obj.co_firstlineno, obj.co_lnotab, obj.co_freevars,
                obj.co_cellvars
            )
        else:
            args = (
                obj.co_argcount, obj.co_nlocals, obj.co_stacksize, obj.co_flags, obj.co_code,
                obj.co_consts, obj.co_names, obj.co_varnames, obj.co_filename, obj.co_name,
                obj.co_firstlineno, obj.co_lnotab, obj.co_freevars, obj.co_cellvars
            )
        self.save_reduce(types.CodeType, args, obj=obj)
    dispatch[types.CodeType] = save_codeobject

    def save_function(self, obj, name=None):
        """ Registered with the dispatch to handle all function types.

        Determines what kind of function obj is (e.g. lambda, defined at
        interactive prompt, etc) and handles the pickling appropriately.
        """
        if obj in _BUILTIN_TYPE_CONSTRUCTORS:
            # We keep a special-cased cache of built-in type constructors at
            # global scope, because these functions are structured very
            # differently in different python versions and implementations (for
            # example, they're instances of types.BuiltinFunctionType in
            # CPython, but they're ordinary types.FunctionType instances in
            # PyPy).
            #
            # If the function we've received is in that cache, we just
            # serialize it as a lookup into the cache.
            return self.save_reduce(_BUILTIN_TYPE_CONSTRUCTORS[obj], (), obj=obj)

        write = self.write

        if name is None:
            name = obj.__name__
        try:
            # whichmodule() could fail, see
            # https://bitbucket.org/gutworth/six/issues/63/importing-six-breaks-pickling
            modname = pickle.whichmodule(obj, name)
        except Exception:
            modname = None
        # print('which gives %s %s %s' % (modname, obj, name))
        try:
            themodule = sys.modules[modname]
        except KeyError:
            # eval'd items such as namedtuple give invalid items for their function __module__
            modname = '__main__'

        if modname == '__main__':
            themodule = None

        try:
            lookedup_by_name = getattr(themodule, name, None)
        except Exception:
            lookedup_by_name = None

        if themodule:
            self.modules.add(themodule)
            if lookedup_by_name is obj:
                return self.save_global(obj, name)

        # a builtin_function_or_method which comes in as an attribute of some
        # object (e.g., itertools.chain.from_iterable) will end
        # up with modname "__main__" and so end up here. But these functions
        # have no __code__ attribute in CPython, so the handling for
        # user-defined functions below will fail.
        # So we pickle them here using save_reduce; have to do it differently
        # for different python versions.
        if not hasattr(obj, '__code__'):
            if PY3:
                rv = obj.__reduce_ex__(self.proto)
            else:
                if hasattr(obj, '__self__'):
                    rv = (getattr, (obj.__self__, name))
                else:
                    raise pickle.PicklingError("Can't pickle %r" % obj)
            return self.save_reduce(obj=obj, *rv)

        # if func is lambda, def'ed at prompt, is in main, or is nested, then
        # we'll pickle the actual function object rather than simply saving a
        # reference (as is done in default pickler), via save_function_tuple.
        if (islambda(obj)
                or getattr(obj.__code__, 'co_filename', None) == '<stdin>'
                or themodule is None):
            self.save_function_tuple(obj)
            return
        else:
            # func is nested
            if lookedup_by_name is None or lookedup_by_name is not obj:
                self.save_function_tuple(obj)
                return

        if obj.__dict__:
            # essentially save_reduce, but workaround needed to avoid recursion
            self.save(_restore_attr)
            write(pickle.MARK + pickle.GLOBAL + modname + '\n' + name + '\n')
            self.memoize(obj)
            self.save(obj.__dict__)
            write(pickle.TUPLE + pickle.REDUCE)
        else:
            write(pickle.GLOBAL + modname + '\n' + name + '\n')
            self.memoize(obj)
    dispatch[types.FunctionType] = save_function

    def _save_subimports(self, code, top_level_dependencies):
        """
        Ensure de-pickler imports any package child-modules that
        are needed by the function
        """
        # check if any known dependency is an imported package
        for x in top_level_dependencies:
            if isinstance(x, types.ModuleType) and hasattr(x, '__package__') and x.__package__:
                # check if the package has any currently loaded sub-imports
                prefix = x.__name__ + '.'
                for name, module in sys.modules.items():
                    # Older versions of pytest will add a "None" module to sys.modules.
                    if name is not None and name.startswith(prefix):
                        # check whether the function can address the sub-module
                        tokens = set(name[len(prefix):].split('.'))
                        if not tokens - set(code.co_names):
                            # ensure unpickler executes this import
                            self.save(module)
                            # then discards the reference to it
                            self.write(pickle.POP)

    def save_dynamic_class(self, obj):
        """
        Save a class that can't be stored as module global.

        This method is used to serialize classes that are defined inside
        functions, or that otherwise can't be serialized as attribute lookups
        from global modules.
        """
        clsdict = dict(obj.__dict__)  # copy dict proxy to a dict
        clsdict.pop('__weakref__', None)

        # On PyPy, __doc__ is a readonly attribute, so we need to include it in
        # the initial skeleton class.  This is safe because we know that the
        # doc can't participate in a cycle with the original class.
        type_kwargs = {'__doc__': clsdict.pop('__doc__', None)}

        # If type overrides __dict__ as a property, include it in the type kwargs.
        # In Python 2, we can't set this attribute after construction.
        __dict__ = clsdict.pop('__dict__', None)
        if isinstance(__dict__, property):
            type_kwargs['__dict__'] = __dict__

        save = self.save
        write = self.write

        # We write pickle instructions explicitly here to handle the
        # possibility that the type object participates in a cycle with its own
        # __dict__. We first write an empty "skeleton" version of the class and
        # memoize it before writing the class' __dict__ itself. We then write
        # instructions to "rehydrate" the skeleton class by restoring the
        # attributes from the __dict__.
        #
        # A type can appear in a cycle with its __dict__ if an instance of the
        # type appears in the type's __dict__ (which happens for the stdlib
        # Enum class), or if the type defines methods that close over the name
        # of the type, (which is common for Python 2-style super() calls).

        # Push the rehydration function.
        save(_rehydrate_skeleton_class)

        # Mark the start of the args tuple for the rehydration function.
        write(pickle.MARK)

        # Create and memoize an skeleton class with obj's name and bases.
        tp = type(obj)
        self.save_reduce(tp, (obj.__name__, obj.__bases__, type_kwargs), obj=obj)

        # Now save the rest of obj's __dict__. Any references to obj
        # encountered while saving will point to the skeleton class.
        save(clsdict)

        # Write a tuple of (skeleton_class, clsdict).
        write(pickle.TUPLE)

        # Call _rehydrate_skeleton_class(skeleton_class, clsdict)
        write(pickle.REDUCE)

    def save_function_tuple(self, func):
        """  Pickles an actual func object.

        A func comprises: code, globals, defaults, closure, and dict.  We
        extract and save these, injecting reducing functions at certain points
        to recreate the func object.  Keep in mind that some of these pieces
        can contain a ref to the func itself.  Thus, a naive save on these
        pieces could trigger an infinite loop of save's.  To get around that,
        we first create a skeleton func object using just the code (this is
        safe, since this won't contain a ref to the func), and memoize it as
        soon as it's created.  The other stuff can then be filled in later.
        """
        if is_tornado_coroutine(func):
            self.save_reduce(_rebuild_tornado_coroutine, (func.__wrapped__,),
                             obj=func)
            return

        save = self.save
        write = self.write

        code, f_globals, defaults, closure_values, dct, base_globals = self.extract_func_data(func)

        save(_fill_function)  # skeleton function updater
        write(pickle.MARK)    # beginning of tuple that _fill_function expects

        self._save_subimports(
            code,
            itertools.chain(f_globals.values(), closure_values or ()),
        )

        # create a skeleton function object and memoize it
        save(_make_skel_func)
        save((
            code,
            len(closure_values) if closure_values is not None else -1,
            base_globals,
        ))
        write(pickle.REDUCE)
        self.memoize(func)

        # save the rest of the func data needed by _fill_function
        state = {
            'globals': f_globals,
            'defaults': defaults,
            'dict': dct,
            'module': func.__module__,
            'closure_values': closure_values,
        }
        if hasattr(func, '__qualname__'):
            state['qualname'] = func.__qualname__
        save(state)
        write(pickle.TUPLE)
        write(pickle.REDUCE)  # applies _fill_function on the tuple

    _extract_code_globals_cache = (
        weakref.WeakKeyDictionary()
        if not hasattr(sys, "pypy_version_info")
        else {})

    @classmethod
    def extract_code_globals(cls, co):
        """
        Find all globals names read or written to by codeblock co
        """
        out_names = cls._extract_code_globals_cache.get(co)
        if out_names is None:
            try:
                names = co.co_names
            except AttributeError:
                # PyPy "builtin-code" object
                out_names = set()
            else:
                out_names = set(names[oparg]
                                for op, oparg in _walk_global_ops(co))

                # see if nested function have any global refs
                if co.co_consts:
                    for const in co.co_consts:
                        if type(const) is types.CodeType:
                            out_names |= cls.extract_code_globals(const)

            cls._extract_code_globals_cache[co] = out_names

        return out_names

    def extract_func_data(self, func):
        """
        Turn the function into a tuple of data necessary to recreate it:
            code, globals, defaults, closure_values, dict
        """
        code = func.__code__

        # extract all global ref's
        func_global_refs = self.extract_code_globals(code)

        # process all variables referenced by global environment
        f_globals = {}
        for var in func_global_refs:
            if var in func.__globals__:
                f_globals[var] = func.__globals__[var]

        # defaults requires no processing
        defaults = func.__defaults__

        # process closure
        closure = (
            list(map(_get_cell_contents, func.__closure__))
            if func.__closure__ is not None
            else None
        )

        # save the dict
        dct = func.__dict__

        base_globals = self.globals_ref.get(id(func.__globals__), {})
        self.globals_ref[id(func.__globals__)] = base_globals

        return (code, f_globals, defaults, closure, dct, base_globals)

    def save_builtin_function(self, obj):
        if obj.__module__ == "__builtin__":
            return self.save_global(obj)
        return self.save_function(obj)
    dispatch[types.BuiltinFunctionType] = save_builtin_function

    def save_global(self, obj, name=None, pack=pack):
        """
        Save a "global".

        The name of this method is somewhat misleading: all types get
        dispatched here.
        """
        if obj.__module__ == "__main__":
            return self.save_dynamic_class(obj)

        try:
            return Pickler.save_global(self, obj, name=name)
        except Exception:
            if obj.__module__ == "__builtin__" or obj.__module__ == "builtins":
                if obj in _BUILTIN_TYPE_NAMES:
                    return self.save_reduce(
                        _builtin_type, (_BUILTIN_TYPE_NAMES[obj],), obj=obj)

            typ = type(obj)
            if typ is not obj and isinstance(obj, (type, types.ClassType)):
                return self.save_dynamic_class(obj)

            raise

    dispatch[type] = save_global
    dispatch[types.ClassType] = save_global

    def save_instancemethod(self, obj):
        # Memoization rarely is ever useful due to python bounding
        if obj.__self__ is None:
            self.save_reduce(getattr, (obj.im_class, obj.__name__))
        else:
            if PY3:
                self.save_reduce(types.MethodType, (obj.__func__, obj.__self__), obj=obj)
            else:
                self.save_reduce(types.MethodType, (obj.__func__, obj.__self__, obj.__self__.__class__),
                         obj=obj)
    dispatch[types.MethodType] = save_instancemethod

    def save_inst(self, obj):
        """Inner logic to save instance. Based off pickle.save_inst"""
        cls = obj.__class__

        # Try the dispatch table (pickle module doesn't do it)
        f = self.dispatch.get(cls)
        if f:
            f(self, obj)  # Call unbound method with explicit self
            return

        memo = self.memo
        write = self.write
        save = self.save

        if hasattr(obj, '__getinitargs__'):
            args = obj.__getinitargs__()
            len(args)  # XXX Assert it's a sequence
            pickle._keep_alive(args, memo)
        else:
            args = ()

        write(pickle.MARK)

        if self.bin:
            save(cls)
            for arg in args:
                save(arg)
            write(pickle.OBJ)
        else:
            for arg in args:
                save(arg)
            write(pickle.INST + cls.__module__ + '\n' + cls.__name__ + '\n')

        self.memoize(obj)

        try:
            getstate = obj.__getstate__
        except AttributeError:
            stuff = obj.__dict__
        else:
            stuff = getstate()
            pickle._keep_alive(stuff, memo)
        save(stuff)
        write(pickle.BUILD)

    if not PY3:
        dispatch[types.InstanceType] = save_inst

    def save_property(self, obj):
        # properties not correctly saved in python
        self.save_reduce(property, (obj.fget, obj.fset, obj.fdel, obj.__doc__), obj=obj)
    dispatch[property] = save_property

    def save_classmethod(self, obj):
        orig_func = obj.__func__
        self.save_reduce(type(obj), (orig_func,), obj=obj)
    dispatch[classmethod] = save_classmethod
    dispatch[staticmethod] = save_classmethod

    def save_itemgetter(self, obj):
        """itemgetter serializer (needed for namedtuple support)"""
        class Dummy:
            def __getitem__(self, item):
                return item
        items = obj(Dummy())
        if not isinstance(items, tuple):
            items = (items, )
        return self.save_reduce(operator.itemgetter, items)

    if type(operator.itemgetter) is type:
        dispatch[operator.itemgetter] = save_itemgetter

    def save_attrgetter(self, obj):
        """attrgetter serializer"""
        class Dummy(object):
            def __init__(self, attrs, index=None):
                self.attrs = attrs
                self.index = index
            def __getattribute__(self, item):
                attrs = object.__getattribute__(self, "attrs")
                index = object.__getattribute__(self, "index")
                if index is None:
                    index = len(attrs)
                    attrs.append(item)
                else:
                    attrs[index] = ".".join([attrs[index], item])
                return type(self)(attrs, index)
        attrs = []
        obj(Dummy(attrs))
        return self.save_reduce(operator.attrgetter, tuple(attrs))

    if type(operator.attrgetter) is type:
        dispatch[operator.attrgetter] = save_attrgetter

    def save_file(self, obj):
        """Save a file"""
        try:
            import StringIO as pystringIO #we can't use cStringIO as it lacks the name attribute
        except ImportError:
            import io as pystringIO

        if not hasattr(obj, 'name') or  not hasattr(obj, 'mode'):
            raise pickle.PicklingError("Cannot pickle files that do not map to an actual file")
        if obj is sys.stdout:
            return self.save_reduce(getattr, (sys,'stdout'), obj=obj)
        if obj is sys.stderr:
            return self.save_reduce(getattr, (sys,'stderr'), obj=obj)
        if obj is sys.stdin:
            raise pickle.PicklingError("Cannot pickle standard input")
        if obj.closed:
            raise pickle.PicklingError("Cannot pickle closed files")
        if hasattr(obj, 'isatty') and obj.isatty():
            raise pickle.PicklingError("Cannot pickle files that map to tty objects")
        if 'r' not in obj.mode and '+' not in obj.mode:
            raise pickle.PicklingError("Cannot pickle files that are not opened for reading: %s" % obj.mode)

        name = obj.name

        retval = pystringIO.StringIO()

        try:
            # Read the whole file
            curloc = obj.tell()
            obj.seek(0)
            contents = obj.read()
            obj.seek(curloc)
        except IOError:
            raise pickle.PicklingError("Cannot pickle file %s as it cannot be read" % name)
        retval.write(contents)
        retval.seek(curloc)

        retval.name = name
        self.save(retval)
        self.memoize(obj)

    def save_ellipsis(self, obj):
        self.save_reduce(_gen_ellipsis, ())

    def save_not_implemented(self, obj):
        self.save_reduce(_gen_not_implemented, ())

    if PY3:
        dispatch[io.TextIOWrapper] = save_file
    else:
        dispatch[file] = save_file

    dispatch[type(Ellipsis)] = save_ellipsis
    dispatch[type(NotImplemented)] = save_not_implemented

    def save_weakset(self, obj):
        self.save_reduce(weakref.WeakSet, (list(obj),))

    dispatch[weakref.WeakSet] = save_weakset

    def save_logger(self, obj):
        self.save_reduce(logging.getLogger, (obj.name,), obj=obj)

    dispatch[logging.Logger] = save_logger

    """Special functions for Add-on libraries"""
    def inject_addons(self):
        """Plug in system. Register additional pickling functions if modules already loaded"""
        pass


# Tornado support

def is_tornado_coroutine(func):
    """
    Return whether *func* is a Tornado coroutine function.
    Running coroutines are not supported.
    """
    if 'tornado.gen' not in sys.modules:
        return False
    gen = sys.modules['tornado.gen']
    if not hasattr(gen, "is_coroutine_function"):
        # Tornado version is too old
        return False
    return gen.is_coroutine_function(func)


def _rebuild_tornado_coroutine(func):
    from tornado import gen
    return gen.coroutine(func)


# Shorthands for legacy support

def dump(obj, file, protocol=None):
    """Serialize obj as bytes streamed into file

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication speed
    between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python.
    """
    CloudPickler(file, protocol=protocol).dump(obj)


def dumps(obj, protocol=None):
    """Serialize obj as a string of bytes allocated in memory

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication speed
    between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python.
    """
    file = StringIO()
    try:
        cp = CloudPickler(file, protocol=protocol)
        cp.dump(obj)
        return file.getvalue()
    finally:
        file.close()


# including pickles unloading functions in this namespace
load = pickle.load
loads = pickle.loads


# hack for __import__ not working as desired
def subimport(name):
    __import__(name)
    return sys.modules[name]


def dynamic_subimport(name, vars):
    mod = imp.new_module(name)
    mod.__dict__.update(vars)
    sys.modules[name] = mod
    return mod


# restores function attributes
def _restore_attr(obj, attr):
    for key, val in attr.items():
        setattr(obj, key, val)
    return obj


def _get_module_builtins():
    return pickle.__builtins__


def print_exec(stream):
    ei = sys.exc_info()
    traceback.print_exception(ei[0], ei[1], ei[2], None, stream)


def _modules_to_main(modList):
    """Force every module in modList to be placed into main"""
    if not modList:
        return

    main = sys.modules['__main__']
    for modname in modList:
        if type(modname) is str:
            try:
                mod = __import__(modname)
            except Exception as e:
                sys.stderr.write('warning: could not import %s\n.  '
                                 'Your function may unexpectedly error due to this import failing;'
                                 'A version mismatch is likely.  Specific error was:\n' % modname)
                print_exec(sys.stderr)
            else:
                setattr(main, mod.__name__, mod)


#object generators:
def _genpartial(func, args, kwds):
    if not args:
        args = ()
    if not kwds:
        kwds = {}
    return partial(func, *args, **kwds)

def _gen_ellipsis():
    return Ellipsis

def _gen_not_implemented():
    return NotImplemented


def _get_cell_contents(cell):
    try:
        return cell.cell_contents
    except ValueError:
        # sentinel used by ``_fill_function`` which will leave the cell empty
        return _empty_cell_value


def instance(cls):
    """Create a new instance of a class.

    Parameters
    ----------
    cls : type
        The class to create an instance of.

    Returns
    -------
    instance : cls
        A new instance of ``cls``.
    """
    return cls()


@instance
class _empty_cell_value(object):
    """sentinel for empty closures
    """
    @classmethod
    def __reduce__(cls):
        return cls.__name__


def _fill_function(*args):
    """Fills in the rest of function data into the skeleton function object

    The skeleton itself is create by _make_skel_func().
    """
    if len(args) == 2:
        func = args[0]
        state = args[1]
    elif len(args) == 5:
        # Backwards compat for cloudpickle v0.4.0, after which the `module`
        # argument was introduced
        func = args[0]
        keys = ['globals', 'defaults', 'dict', 'closure_values']
        state = dict(zip(keys, args[1:]))
    elif len(args) == 6:
        # Backwards compat for cloudpickle v0.4.1, after which the function
        # state was passed as a dict to the _fill_function it-self.
        func = args[0]
        keys = ['globals', 'defaults', 'dict', 'module', 'closure_values']
        state = dict(zip(keys, args[1:]))
    else:
        raise ValueError('Unexpected _fill_value arguments: %r' % (args,))

    func.__globals__.update(state['globals'])
    func.__defaults__ = state['defaults']
    func.__dict__ = state['dict']
    if 'module' in state:
        func.__module__ = state['module']
    if 'qualname' in state:
        func.__qualname__ = state['qualname']

    cells = func.__closure__
    if cells is not None:
        for cell, value in zip(cells, state['closure_values']):
            if value is not _empty_cell_value:
                cell_set(cell, value)

    return func


def _make_empty_cell():
    if False:
        # trick the compiler into creating an empty cell in our lambda
        cell = None
        raise AssertionError('this route should not be executed')

    return (lambda: cell).__closure__[0]


def _make_skel_func(code, cell_count, base_globals=None):
    """ Creates a skeleton function object that contains just the provided
        code and the correct number of cells in func_closure.  All other
        func attributes (e.g. func_globals) are empty.
    """
    if base_globals is None:
        base_globals = {}
    base_globals['__builtins__'] = __builtins__

    closure = (
        tuple(_make_empty_cell() for _ in range(cell_count))
        if cell_count >= 0 else
        None
    )
    return types.FunctionType(code, base_globals, None, None, closure)


def _rehydrate_skeleton_class(skeleton_class, class_dict):
    """Put attributes from `class_dict` back on `skeleton_class`.

    See CloudPickler.save_dynamic_class for more info.
    """
    for attrname, attr in class_dict.items():
        setattr(skeleton_class, attrname, attr)
    return skeleton_class


def _find_module(mod_name):
    """
    Iterate over each part instead of calling imp.find_module directly.
    This function is able to find submodules (e.g. sickit.tree)
    """
    path = None
    for part in mod_name.split('.'):
        if path is not None:
            path = [path]
        file, path, description = imp.find_module(part, path)
        if file is not None:
            file.close()
    return path, description

"""Constructors for 3rd party libraries
Note: These can never be renamed due to client compatibility issues"""

def _getobject(modname, attribute):
    mod = __import__(modname, fromlist=[attribute])
    return mod.__dict__[attribute]


""" Use copy_reg to extend global pickle definitions """

if sys.version_info < (3, 4):
    method_descriptor = type(str.upper)

    def _reduce_method_descriptor(obj):
        return (getattr, (obj.__objclass__, obj.__name__))

    try:
        import copy_reg as copyreg
    except ImportError:
        import copyreg
    copyreg.pickle(method_descriptor, _reduce_method_descriptor)
