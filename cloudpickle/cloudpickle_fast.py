"""
New, fast version of the Cloudpickler.

This new Cloudpickler class can now extend the fast C Pickler instead of the
previous pythonic Pickler. Because this functionality is only available for
python versions 3.8+, a lot of backward-compatibilty code is also removed.
"""
import abc
import io
import itertools
import logging
import _pickle
import pickle
import sys
import types
import weakref

from _pickle import Pickler

from .cloudpickle import (
    _is_dynamic, _extract_code_globals, _BUILTIN_TYPE_NAMES, DEFAULT_PROTOCOL,
    _find_loaded_submodules, _get_cell_contents, _is_global, _builtin_type,
    Enum, _ensure_tracking, _lookup_class_or_track, _make_skeleton_class,
    _make_skeleton_enum, _extract_class_dict, string_types
)

load, loads = _pickle.load, _pickle.loads

# Shorthands similar to pickle.dump/pickle.dumps


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
    with io.BytesIO() as file:
        cp = CloudPickler(file, protocol=protocol)
        cp.dump(obj)
        return file.getvalue()

# COLLECTION OF OBJECTS __getnewargs__-LIKE METHODS
# -------------------------------------------------


def _function_getnewargs(func, globals_ref):
    code = func.__code__

    # base_globals represents the future global namespace of func at
    # unpickling time. Looking it up and storing it in globals_ref allow
    # functions sharing the same globals at pickling time to also
    # share them once unpickled, at one condition: since globals_ref is
    # an attribute of a Cloudpickler instance, and that a new CloudPickler is
    # created each time pickle.dump or pickle.dumps is called, functions
    # also need to be saved within the same invokation of
    # cloudpickle.dump/cloudpickle.dumps
    # (for example: cloudpickle.dumps([f1, f2])). There
    # is no such limitation when using Cloudpickler.dump, as long as the
    # multiple invokations are bound to the same Cloudpickler.
    base_globals = globals_ref.setdefault(id(func.__globals__), {})

    if base_globals == {}:
        # Add module attributes used to resolve relative imports
        # instructions inside func.
        for k in ["__package__", "__name__", "__path__", "__file__"]:
            # Some built-in functions/methods such as object.__new__  have
            # their __globals__ set to None in PyPy
            if func.__globals__ is not None and k in func.__globals__:
                base_globals[k] = func.__globals__[k]

    # Do not bind the free variables before the function is created to avoid
    # infinite recursion.
    if func.__closure__ is None:
        closure = None
    else:
        closure = tuple(types.CellType() for _ in range(len(code.co_freevars)))

    return code, base_globals, None, None, closure


def _class_getnewargs(obj):
    # On PyPy, __doc__ is a readonly attribute, so we need to include it in
    # the initial skeleton class.  This is safe because we know that the
    # doc can't participate in a cycle with the original class.
    type_kwargs = {'__doc__': obj.__dict__.get('__doc__', None)}

    if hasattr(obj, "__slots__"):
        type_kwargs["__slots__"] = obj.__slots__

    __dict__ = obj.__dict__.get('__dict__', None)
    if isinstance(__dict__, property):
        type_kwargs['__dict__'] = __dict__

    return (type(obj), obj.__name__, obj.__bases__, type_kwargs,
            _ensure_tracking(obj), None)


def _enum_getnewargs(obj):
    members = dict((e.name, e.value) for e in obj)
    return (obj.__bases__, obj.__name__, obj.__qualname__, members,
            obj.__module__, _ensure_tracking(obj), None)


# COLLECTION OF OBJECTS RECONSTRUCTORS
# ------------------------------------

# Builtin types are types defined in the python language source code, that are
# not defined in an importable python module (Lib/* for pure python module,
# Modules/* for C-implemented modules). The most wildely used ones (such as
# tuple, dict, list) are made accessible in any interpreter session by exposing
# them in the builtin namespace at startup time.

# By construction, builtin types do not have a module. Trying to access their
# __module__ attribute will default to 'builtins', that only contains builtin
# types accessible at interpreter startup. Therefore, trying to pickle the
# other ones using classic module attribute lookup instructions will fail.

# Fortunately, many non-accessible builtin-types are mirrored in the types
# module. For those types, we pickle the function builtin_type_reconstructor
# instead, that contains instruction to look them up via the types module.
def _builtin_type_reconstructor(name):
    """Return a builtin-type using attribute lookup from the types module"""
    return getattr(types, name)


# XXX: what does "not working as desired" means?
# hack for __import__ not working as desired
def _module_reconstructor(name):
    __import__(name)
    return sys.modules[name]


def _dynamic_module_reconstructor(name, vars):
    mod = types.ModuleType(name)
    mod.__dict__.update(vars)
    return mod


def _file_reconstructor(retval):
    return retval


# COLLECTION OF OBJECTS STATE GETTERS
# -----------------------------------
def _function_getstate(func):
    # - Put func's dynamic attributes (stored in func.__dict__) in state. These
    #   attributes will be restored at unpickling time using
    #   f.__dict__.update(state)
    # - Put func's members into slotstate. Such attributes will be restored at
    #   unpickling time by iterating over slotstate and calling setattr(func,
    #   slotname, slotvalue)
    slotstate = {
        "__name__": func.__name__,
        "__qualname__": func.__qualname__,
        "__annotations__": func.__annotations__,
        "__kwdefaults__": func.__kwdefaults__,
        "__defaults__": func.__defaults__,
        "__module__": func.__module__,
        "__doc__": func.__doc__,
        "__closure__": func.__closure__,
    }

    f_globals_ref = _extract_code_globals(func.__code__)
    f_globals = {k: func.__globals__[k] for k in f_globals_ref if k in
                 func.__globals__}

    closure_values = (
        list(map(_get_cell_contents, func.__closure__))
        if func.__closure__ is not None else ()
    )

    # extract submodules referenced by attribute lookup (no global opcode)
    f_globals["__submodules__"] = _find_loaded_submodules(
        func.__code__, itertools.chain(f_globals.values(), closure_values))
    slotstate["__globals__"] = f_globals

    state = func.__dict__
    return state, slotstate


def _class_getstate(obj):
    clsdict = _extract_class_dict(obj)
    clsdict.pop('__weakref__', None)
    clsdict.pop('__doc__', None)  # present in the reconstructor args

    # For ABCMeta in python3.7+, remove _abc_impl as it is not picklable.
    # This is a fix which breaks the cache but this only makes the first
    # calls to issubclass slower.
    if "_abc_impl" in clsdict:
        (registry, _, _, _) = abc._get_dump(obj)
        clsdict["_abc_impl"] = [subclass_weakref()
                                for subclass_weakref in registry]
    if hasattr(obj, "__slots__"):
        # pickle string length optimization: member descriptors of obj are
        # created automatically from obj's __slots__ attribute, no need to
        # save them in obj's state
        if isinstance(obj.__slots__, string_types):
            clsdict.pop(obj.__slots__)
        else:
            for k in obj.__slots__:
                clsdict.pop(k, None)

    clsdict.pop('__dict__', None)  # unpickleable property object

    return (clsdict, {})


def _enum_getstate(obj):
    clsdict, slotstate = _class_getstate(obj)

    members = dict((e.name, e.value) for e in obj)
    # Cleanup the clsdict that will be passed to _rehydrate_skeleton_class:
    # Those attributes are already handled by the metaclass.
    for attrname in ["_generate_next_value_", "_member_names_",
                     "_member_map_", "_member_type_",
                     "_value2member_map_"]:
        clsdict.pop(attrname, None)
    for member in members:
        clsdict.pop(member)
        # Special handling of Enum subclasses
    return clsdict, slotstate


# COLLECTIONS OF OBJECTS REDUCERS
# -------------------------------
# A reducer is a function taking a single argument (obj), and that returns a
# tuple with all the necessary data to re-construct obj. Apart from a few
# exceptions (list, dicts, bytes, ints, etc.), a reducer is necessary to
# correclty pickle an object.
# While many built-in objects (Exceptions objects, instances of the "object"
# class, etc), are shipped with their own built-in reducer (invoked using
# obj.__reduce__), some do not. The following methods were created to "fill
# these holes".

# XXX: no itemgetter/attrgetter reducer support implemented as the tests seem
# to pass even without them


def _builtin_type_reduce(obj):
    return _builtin_type_reconstructor, (_BUILTIN_TYPE_NAMES[obj],)


def _code_reduce(obj):
    """codeobject reducer"""
    if hasattr(obj, "co_posonlyargcount"):  # pragma: no branch
        args = (
            obj.co_argcount, obj.co_posonlyargcount,
            obj.co_kwonlyargcount, obj.co_nlocals, obj.co_stacksize,
            obj.co_flags, obj.co_code, obj.co_consts, obj.co_names,
            obj.co_varnames, obj.co_filename, obj.co_name,
            obj.co_firstlineno, obj.co_lnotab, obj.co_freevars,
            obj.co_cellvars
        )
    else:
        args = (
            obj.co_argcount, obj.co_kwonlyargcount, obj.co_nlocals,
            obj.co_stacksize, obj.co_flags, obj.co_code, obj.co_consts,
            obj.co_names, obj.co_varnames, obj.co_filename,
            obj.co_name, obj.co_firstlineno, obj.co_lnotab,
            obj.co_freevars, obj.co_cellvars
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


def _mappingproxy_reduce(obj):
    return types.MappingProxyType, (dict(obj),)


def _memoryview_reduce(obj):
    return bytes, (obj.tobytes(),)


def _module_reduce(obj):
    if _is_dynamic(obj):
        return _dynamic_module_reconstructor, (obj.__name__, vars(obj))
    else:
        return _module_reconstructor, (obj.__name__,)


def _method_reduce(obj):
    return (types.MethodType, (obj.__func__, obj.__self__))


def _logger_reduce(obj):
    return logging.getLogger, (obj.name,)


def _root_logger_reduce(obj):
    return logging.getLogger, ()


def _weakset_reduce(obj):
    return weakref.WeakSet, (list(obj),)


def _dynamic_function_reduce(func, globals_ref):
    """Reduce a function that is not pickleable via attribute loookup.
    """
    newargs = _function_getnewargs(func, globals_ref)
    state = _function_getstate(func)
    return types.FunctionType, newargs, state, None, None, _function_setstate


def _function_reduce(obj, globals_ref):
    """Select the reducer depending on obj's dynamic nature

    This functions starts by replicating save_global: trying to retrieve obj
    from an attribute lookup of a file-backed module. If this check fails, then
    a custom reducer is called.
    """
    if not _is_global(obj):
        return _dynamic_function_reduce(obj, globals_ref)
    return NotImplemented


def _dynamic_class_reduce(obj):
    """
    Save a class that can't be stored as module global.

    This method is used to serialize classes that are defined inside
    functions, or that otherwise can't be serialized as attribute lookups
    from global modules.
    """
    if Enum is not None and issubclass(obj, Enum):
        return (
            _make_skeleton_enum, _enum_getnewargs(obj), _enum_getstate(obj),
            None, None, _class_setstate
        )
    else:
        return (
            _make_skeleton_class, _class_getnewargs(obj), _class_getstate(obj),
            None, None, _class_setstate
        )


def _class_reduce(obj):
    """Select the reducer depending on the dynamic nature of the class obj"""
    # XXX: there used to be special handling for NoneType, EllipsisType and
    # NotImplementedType. As for now this module handles only python3.8+, this
    # code has been removed.
    if obj is type(None):  # noqa
        return type, (None,)
    elif obj is type(Ellipsis):
        return type, (Ellipsis,)
    elif obj is type(NotImplemented):
        return type, (NotImplemented,)
    elif obj in _BUILTIN_TYPE_NAMES:
        return _builtin_type, (_BUILTIN_TYPE_NAMES[obj],)
    elif not _is_global(obj):
        return _dynamic_class_reduce(obj)
    return NotImplemented


# COLLECTIONS OF OBJECTS STATE SETTERS
# ------------------------------------
# state setters are called at unpickling time, once the object is created and
# it has to be updated to how it was at unpickling time.


def _function_setstate(obj, state):
    """Update the state of a dynaamic function.

    As __closure__ and __globals__ are readonly attributes of a function, we
    cannot rely on the native setstate routine of pickle.load_build, that calls
    setattr on items of the slotstate. Instead, we have to modify them inplace.
    """
    state, slotstate = state
    obj.__dict__.update(state)

    obj_globals = slotstate.pop("__globals__")
    obj_closure = slotstate.pop("__closure__")

    # remove uncessary  references to submodules
    obj_globals.pop("__submodules__")
    obj.__globals__.update(obj_globals)
    obj.__globals__["__builtins__"] = __builtins__

    if obj_closure is not None:
        for i, cell in enumerate(obj_closure):
            try:
                value = cell.cell_contents
            except ValueError:  # cell is empty
                continue
            obj.__closure__[i].cell_contents = value

    for k, v in slotstate.items():
        setattr(obj, k, v)


def _class_setstate(obj, state):
    state, slotstate = state
    registry = None
    for attrname, attr in state.items():
        if attrname == "_abc_impl":
            registry = attr
        else:
            setattr(obj, attrname, attr)
    if registry is not None:
        for subclass in registry:
            obj.register(subclass)

    return obj


class CloudPickler(Pickler):
    """Fast C Pickler extension with additional reducing routines.

    Cloudpickler's extensions exist into into:

    * it's dispatch_table containing reducers that are called only if ALL
      built-in saving functions were previously discarded.
    * a special callback, invoked before standard function/class
      builtin-saving method (save_global), to serialize dynamic functions
    """

    dispatch = {}
    dispatch[classmethod] = _classmethod_reduce
    dispatch[io.TextIOWrapper] = _file_reduce
    dispatch[logging.Logger] = _logger_reduce
    dispatch[logging.RootLogger] = _root_logger_reduce
    dispatch[memoryview] = _memoryview_reduce
    dispatch[staticmethod] = _classmethod_reduce
    dispatch[types.CellType] = _cell_reduce
    dispatch[types.CodeType] = _code_reduce
    dispatch[types.ModuleType] = _module_reduce
    dispatch[types.MethodType] = _method_reduce
    dispatch[types.MappingProxyType] = _mappingproxy_reduce
    dispatch[weakref.WeakSet] = _weakset_reduce

    def __init__(self, file, protocol=None):
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        Pickler.__init__(self, file, protocol=protocol)
        # map functions __globals__ attribute ids, to ensure that functions
        # sharing the same global namespace at pickling time also share their
        # global namespace at unpickling time.
        self.globals_ref = {}
        self.dispatch_table = self.dispatch

        # Pickling functions and classes cannot be customized using the
        # dispatch_table: indeed, pickling an object using the dispatch_table
        # works by invoking a reducer specific to the object's type. When the
        # object is a class, its type is often ``type``, except when the class
        # is an instance of another metaclass. In this cased, the metaclass
        # will likely not be known in advance, and thus cannot be special-cased
        # using an entry in the dispatch_table.

        # The pickler's reducer_override, among other things, allows us to
        # register a reducer that will be called for any class, independently
        # of its type.
        self.proto = int(protocol)

    def reducer_override(self, obj):
        """Custom reducing callback for functions and classes

        This function is the analog of a custom save_global. However, the C
        Pickler API does not expose low-level instructions such as save or
        write. Instead, we return a reduce value the the Pickler will
        internally serialize via save_reduce.
        """
        t = type(obj)
        try:
            is_anyclass = issubclass(t, type)
        except TypeError:  # t is not a class (old Boost; see SF #502085)
            is_anyclass = False

        if is_anyclass:
            return _class_reduce(obj)
        elif isinstance(obj, types.FunctionType):
            return _function_reduce(obj, self.globals_ref)
        else:
            # fallback to save_global, including the pickler's distpatch_table
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
