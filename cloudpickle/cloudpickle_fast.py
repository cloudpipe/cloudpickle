"""
New, fast version of the CloudPickler.

This new CloudPickler class can now extend the fast C Pickler instead of the
previous Python implementation of the Pickler class. Because this functionality
is only available for Python versions 3.8+, a lot of backward-compatibility
code is also removed.

Note that the C Pickler sublassing API is CPython-specific. Therefore, some
guards present in cloudpickle.py that were written to handle PyPy specificities
are not present in cloudpickle_fast.py
"""
import abc
import copyreg
import io
import itertools
import logging
import _pickle
import pickle
import sys
import types
import weakref
import typing

from _pickle import Pickler

from .cloudpickle import (
    _is_dynamic, _extract_code_globals, _BUILTIN_TYPE_NAMES, DEFAULT_PROTOCOL,
    _find_imported_submodules, _get_cell_contents, _is_importable_by_name, _builtin_type,
    Enum, _get_or_create_tracker_id,  _make_skeleton_class, _make_skeleton_enum,
    _extract_class_dict, dynamic_subimport, subimport, _typevar_reduce, _get_bases,
    FunctionSaverMixin, _class_reduce
)

load, loads = _pickle.load, _pickle.loads


# Shorthands similar to pickle.dump/pickle.dumps
def dump(obj, file, protocol=None, buffer_callback=None):
    """Serialize obj as bytes streamed into file

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication speed
    between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python.
    """
    CloudPickler(file, protocol=protocol, buffer_callback=buffer_callback).dump(obj)


def dumps(obj, protocol=None, buffer_callback=None):
    """Serialize obj as a string of bytes allocated in memory

    protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
    pickle.HIGHEST_PROTOCOL. This setting favors maximum communication speed
    between processes running the same Python version.

    Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
    compatibility with older versions of Python.
    """
    with io.BytesIO() as file:
        cp = CloudPickler(file, protocol=protocol, buffer_callback=buffer_callback)
        cp.dump(obj)
        return file.getvalue()


# COLLECTION OF OBJECTS __getnewargs__-LIKE METHODS
# -------------------------------------------------

# COLLECTION OF OBJECTS RECONSTRUCTORS
# ------------------------------------
def _file_reconstructor(retval):
    return retval


# COLLECTION OF OBJECTS STATE GETTERS
# -----------------------------------
# COLLECTIONS OF OBJECTS REDUCERS
# -------------------------------
# A reducer is a function taking a single argument (obj), and that returns a
# tuple with all the necessary data to re-construct obj. Apart from a few
# exceptions (list, dict, bytes, int, etc.), a reducer is necessary to
# correctly pickle an object.
# While many built-in objects (Exceptions objects, instances of the "object"
# class, etc), are shipped with their own built-in reducer (invoked using
# obj.__reduce__), some do not. The following methods were created to "fill
# these holes".

def _code_reduce(obj):
    """codeobject reducer"""
    args = (
        obj.co_argcount, obj.co_posonlyargcount,
        obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
        obj.co_flags, obj.co_code, obj.co_consts, obj.co_names,
        obj.co_varnames, obj.co_filename, obj.co_name,
        obj.co_firstlineno, obj.co_lnotab, obj.co_freevars,
        obj.co_cellvars
    )
    return types.CodeType, args


def _cell_reduce(obj):
    """Cell (containing values of a function's free variables) reducer"""
    try:
        obj.cell_contents
    except ValueError:  # cell is empty
        return types.CellType, ()
    else:
        return types.CellType, (obj.cell_contents,)


def _classmethod_reduce(obj):
    orig_func = obj.__func__
    return type(obj), (orig_func,)


def _file_reduce(obj):
    """Save a file"""
    import io

    if not hasattr(obj, "name") or not hasattr(obj, "mode"):
        raise pickle.PicklingError(
            "Cannot pickle files that do not map to an actual file"
        )
    if obj is sys.stdout:
        return getattr, (sys, "stdout")
    if obj is sys.stderr:
        return getattr, (sys, "stderr")
    if obj is sys.stdin:
        raise pickle.PicklingError("Cannot pickle standard input")
    if obj.closed:
        raise pickle.PicklingError("Cannot pickle closed files")
    if hasattr(obj, "isatty") and obj.isatty():
        raise pickle.PicklingError(
            "Cannot pickle files that map to tty objects"
        )
    if "r" not in obj.mode and "+" not in obj.mode:
        raise pickle.PicklingError(
            "Cannot pickle files that are not opened for reading: %s"
            % obj.mode
        )

    name = obj.name

    retval = io.StringIO()

    try:
        # Read the whole file
        curloc = obj.tell()
        obj.seek(0)
        contents = obj.read()
        obj.seek(curloc)
    except IOError:
        raise pickle.PicklingError(
            "Cannot pickle file %s as it cannot be read" % name
        )
    retval.write(contents)
    retval.seek(curloc)

    retval.name = name
    return _file_reconstructor, (retval,)


def _getset_descriptor_reduce(obj):
    return getattr, (obj.__objclass__, obj.__name__)


def _mappingproxy_reduce(obj):
    return types.MappingProxyType, (dict(obj),)


def _memoryview_reduce(obj):
    return bytes, (obj.tobytes(),)


def _module_reduce(obj):
    if _is_dynamic(obj):
        obj.__dict__.pop('__builtins__', None)
        return dynamic_subimport, (obj.__name__, vars(obj))
    else:
        return subimport, (obj.__name__,)


def _method_reduce(obj):
    return (types.MethodType, (obj.__func__, obj.__self__))


def _logger_reduce(obj):
    return logging.getLogger, (obj.name,)


def _root_logger_reduce(obj):
    return logging.getLogger, ()


def _property_reduce(obj):
    return property, (obj.fget, obj.fset, obj.fdel, obj.__doc__)


def _weakset_reduce(obj):
    return weakref.WeakSet, (list(obj),)


# COLLECTIONS OF OBJECTS STATE SETTERS
# ------------------------------------
# state setters are called at unpickling time, once the object is created and
# it has to be updated to how it was at unpickling time.


class CloudPickler(FunctionSaverMixin, Pickler):
    """Fast C Pickler extension with additional reducing routines.

    CloudPickler's extensions exist into into:

    * its dispatch_table containing reducers that are called only if ALL
      built-in saving functions were previously discarded.
    * a special callback named "reducer_override", invoked before standard
      function/class builtin-saving method (save_global), to serialize dynamic
      functions
    """

    # cloudpickle's own dispatch_table, containing the additional set of
    # objects (compared to the standard library pickle) that cloupickle can
    # serialize.
    dispatch = {}
    dispatch[classmethod] = _classmethod_reduce
    dispatch[io.TextIOWrapper] = _file_reduce
    dispatch[logging.Logger] = _logger_reduce
    dispatch[logging.RootLogger] = _root_logger_reduce
    dispatch[memoryview] = _memoryview_reduce
    dispatch[property] = _property_reduce
    dispatch[staticmethod] = _classmethod_reduce
    dispatch[types.CellType] = _cell_reduce
    dispatch[types.CodeType] = _code_reduce
    dispatch[types.GetSetDescriptorType] = _getset_descriptor_reduce
    dispatch[types.ModuleType] = _module_reduce
    dispatch[types.MethodType] = _method_reduce
    dispatch[types.MappingProxyType] = _mappingproxy_reduce
    dispatch[weakref.WeakSet] = _weakset_reduce
    dispatch[typing.TypeVar] = _typevar_reduce

    def __init__(self, file, protocol=None, buffer_callback=None):
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        Pickler.__init__(self, file, protocol=protocol, buffer_callback=buffer_callback)
        # map functions __globals__ attribute ids, to ensure that functions
        # sharing the same global namespace at pickling time also share their
        # global namespace at unpickling time.
        self.globals_ref = {}

        # Take into account potential custom reducers registered by external
        # modules
        self.dispatch_table = copyreg.dispatch_table.copy()
        self.dispatch_table.update(self.dispatch)
        self.proto = int(protocol)

    def reducer_override(self, obj):
        """Type-agnostic reducing callback for function and classes.

        For performance reasons, subclasses of the C _pickle.Pickler class
        cannot register custom reducers for functions and classes in the
        dispatch_table. Reducer for such types must instead implemented in the
        special reducer_override method.

        Note that method will be called for any object except a few
        builtin-types (int, lists, dicts etc.), which differs from reducers in
        the Pickler's dispatch_table, each of them being invoked for objects of
        a specific type only.

        This property comes in handy for classes: although most classes are
        instances of the ``type`` metaclass, some of them can be instances of
        other custom metaclasses (such as enum.EnumMeta for example). In
        particular, the metaclass will likely not be known in advance, and thus
        cannot be special-cased using an entry in the dispatch_table.
        reducer_override, among other things, allows us to register a reducer
        that will be called for any class, independently of its type.


        Notes:

        * reducer_override has the priority over dispatch_table-registered
          reducers.
        * reducer_override can be used to fix other limitations of cloudpickle
          for other types that suffered from type-specific reducers, such as
          Exceptions. See https://github.com/cloudpipe/cloudpickle/issues/248
        """
        t = type(obj)
        try:
            is_anyclass = issubclass(t, type)
        except TypeError:  # t is not a class (old Boost; see SF #502085)
            is_anyclass = False

        if is_anyclass:
            return _class_reduce(obj)
        elif isinstance(obj, types.FunctionType):
            return self._function_reduce(obj)
        else:
            # fallback to save_global, including the Pickler's distpatch_table
            return NotImplemented

    def dump(self, obj):
        try:
            return Pickler.dump(self, obj)
        except RuntimeError as e:
            if "recursion" in e.args[0]:
                msg = (
                    "Could not pickle object as excessively deep recursion "
                    "required."
                )
                raise pickle.PicklingError(msg)
            else:
                raise
