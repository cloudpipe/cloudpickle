"""
New, fast version of the Cloudpickler.

This new Cloudpickler class can now extend the fast C Pickler instead of the
previous pythonic Pickler. Because this functionality is only available for
python versions 3.8+, a lot of backward-compatibilty code is also removed.
"""
import abc
import dis
import io
import itertools
import logging
import opcode
import _pickle
import pickle
import sys
import types
import weakref

from _pickle import Pickler

from .cloudpickle import (
    islambda, _is_dynamic, extract_code_globals, GLOBAL_OPS,
    _BUILTIN_TYPE_CONSTRUCTORS, _BUILTIN_TYPE_NAMES, DEFAULT_PROTOCOL,
    _find_loaded_submodules, _get_cell_contents
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
    file = io.BytesIO()
    try:
        cp = CloudPickler(file, protocol=protocol)
        cp.dump(obj)
        return file.getvalue()
    finally:
        file.close()


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

    # Do not bind the free variables before the function is created to avoid
    # infinite recursion.
    if func.__closure__ is None:
        closure = None
    else:
        closure = tuple(types.CellType() for _ in range(len(code.co_freevars)))

    return code, base_globals, None, None, closure


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

    f_globals_ref = extract_code_globals(func.__code__)
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
    args = (
        obj.co_argcount,
        obj.co_kwonlyargcount,
        obj.co_nlocals,
        obj.co_stacksize,
        obj.co_flags,
        obj.co_code,
        obj.co_consts,
        obj.co_names,
        obj.co_varnames,
        obj.co_filename,
        obj.co_name,
        obj.co_firstlineno,
        obj.co_lnotab,
        obj.co_freevars,
        obj.co_cellvars,
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
    # XXX: should globals_ref be a global variable instead? The reason is
    # purely cosmetic.
    newargs = _function_getnewargs(func, globals_ref)
    state = _function_getstate(func)
    return types.FunctionType, newargs, state, None, None, _function_setstate


def _function_reduce(obj, globals_ref):
    """Select the reducer depending on obj's dynamic nature

    This functions starts by replicating save_global: trying to retrieve obj
    from an attribute lookup of a file-backed module. If this check fails, then
    a custom reducer is called.
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
        return _BUILTIN_TYPE_CONSTRUCTORS[obj], ()

    name = obj.__name__
    try:
        modname = pickle.whichmodule(obj, name)
    except Exception:
        modname = None

    try:
        themodule = sys.modules[modname]
    except KeyError:
        # eval'd items such as namedtuple give invalid items for their function
        # __module__
        modname = "__main__"

    if modname == "__main__":
        # we do not want the incoming module attribute lookup to succeed for
        # the __main__ module.
        themodule = None

    try:
        lookedup_by_name = getattr(themodule, name, None)
    except Exception:
        lookedup_by_name = None

    if lookedup_by_name is obj:  # in this case, module is None
        # if obj exists in a filesytem-backed module, let the builtin pickle
        # saving routines save obj
        return NotImplementedError

    # XXX: the special handling of builtin_function_or_method is removed as
    # currently this hook is not called for such instances, as opposed to
    # cloudpickle.

    # if func is lambda, def'ed at prompt, is in main, or is nested, then
    # we'll pickle the actual function object rather than simply saving a
    # reference (as is done in default pickler), via save_function_tuple.
    if (
        islambda(obj)
        or getattr(obj.__code__, "co_filename", None) == "<stdin>"
        or themodule is None
    ):
        return _dynamic_function_reduce(obj, globals_ref=globals_ref)

    # this whole code section may be cleanable: the if/else conditions + the
    # NotImplementedError look like they cover nearly all cases.
    else:
        # func is nested
        if lookedup_by_name is None or lookedup_by_name is not obj:
            return _dynamic_function_reduce(obj, globals_ref=globals_ref)


def _dynamic_class_reduce(obj):
    """
    Save a class that can't be stored as module global.

    This method is used to serialize classes that are defined inside
    functions, or that otherwise can't be serialized as attribute lookups
    from global modules.
    """
    # XXX: This code is nearly untouch with regards to the legacy cloudpickle.
    # It is pretty and hard to understand. Maybe refactor it by dumping
    # potential python2 specific code and making a trading off optimizations in
    # favor of readbility.
    clsdict = dict(obj.__dict__)  # copy dict proxy to a dict
    clsdict.pop("__weakref__", None)

    # XXX: I am trying to add the abc-registered subclasses into the class
    # reconstructor, because using save_reduce semantics prevents us to perform
    # any other operation than state updating after obj is created.

    # I may encounter reference cycles, although there seems to be checks
    # preventing this to happen.
    if "_abc_impl" in clsdict:
        (registry, _, _, _) = abc._get_dump(obj)
        clsdict["_abc_impl"] = [
            subclass_weakref() for subclass_weakref in registry
        ]

    # On PyPy, __doc__ is a readonly attribute, so we need to include it in
    # the initial skeleton class.  This is safe because we know that the
    # doc can't participate in a cycle with the original class.
    type_kwargs = {"__doc__": clsdict.pop("__doc__", None)}

    if hasattr(obj, "__slots__"):
        type_kwargs["__slots__"] = obj.__slots__
        # Pickle string length optimization: member descriptors of obj are
        # created automatically from obj's __slots__ attribute, no need to
        # save them in obj's state
        if isinstance(obj.__slots__, str):
            clsdict.pop(obj.__slots__)
        else:
            for k in obj.__slots__:
                clsdict.pop(k, None)

    # If type overrides __dict__ as a property, include it in the type kwargs.
    # In Python 2, we can't set this attribute after construction.
    # XXX: removed special handling of __dict__ for python2
    __dict__ = clsdict.pop("__dict__", None)
    if isinstance(__dict__, property):
        type_kwargs["__dict__"] = __dict__
        __dict__ = None

    return (
        type(obj),
        (obj.__name__, obj.__bases__, type_kwargs),
        (clsdict, {}),
        None,
        None,
        _class_setstate,
    )


def _class_reduce(obj):
    """Select the reducer depending on the dynamic nature of the class obj"""
    # XXX: there used to be special handling for NoneType, EllipsisType and
    # NotImplementedType. As for now this module handles only python3.8+, this
    # code has been removed.
    if obj.__module__ == "__main__":
        return _dynamic_class_reduce(obj)

    try:
        # All classes are caught in this function: pickleable classes are
        # filtered out by creating a Pickler with no custom class reducer
        # (thus, falling back to save_global). If it fails to save obj, then
        # obj is either a non-pickleable builtin or dynamic.
        pickle.dumps(obj)
    except Exception:
        # XXX: previously, we also looked for the __builtin__ module, but this
        # is python 2 specific.
        if obj.__module__ == "builtins":
            if obj in _BUILTIN_TYPE_NAMES:
                return _builtin_type_reduce(obj)

        typ = type(obj)
        if typ is not obj and isinstance(obj, type):  # noqa: E721
            return _dynamic_class_reduce(obj)

    else:
        # if pickle.dumps worked out fine, then simply fallback to the
        # traditional pickle by attribute # implemented in the builtin
        # `Pickler.save_global`.
        return NotImplementedError


# COLLECTIONS OF OBJECTS STATE SETTERS
# ------------------------------------
# state setters are called at unpickling time, once the object is created and
# it has to be updated to how it was at unpickling time.


def _function_setstate(obj, state, slotstate):
    """Update the state of a dynaamic function.

    As __closure__ and __globals__ are readonly attributes of a function, we
    cannot rely on the native setstate routine of pickle.load_build, that calls
    setattr on items of the slotstate. Instead, we have to modify them inplace.
    """
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


def _class_setstate(obj, state, slotstate):
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


# Arbitration between builtin-save method and user-defined callbacks
# ------------------------------------------------------------------
# This set of functions aim at deciding whether an object can be properly
# pickler by the c Pickler, or if it needs to be serialized using cloudpickle's
# reducers.
def _reduce_global(pickler, obj):
    """Custom reducing callback for functions and classes

    This function is the analog of a custom save_global. However, the C Pickler
    API does not expose low-level instructions such as save or write. Instead,
    we return a reduce value the the Pickler will internally serialize via
    save_reduce.
    """
    # Classes deriving from custom, dynamic metaclasses won't get caught inside
    # the hook_dispatch dict. In the legacy cloudpickle, this was not really a
    # problem because not being present in the dispatch table meant falling
    # back to save_global, which was already overriden by cloudpickle. Using
    # the c pickler, save_global cannot be overriden, so we have manually check
    # is obj's comes from a custom metaclass, and in this case, direct the
    # object to save_global.
    t = type(obj)

    try:
        is_metaclass = issubclass(t, type)
    except TypeError:  # t is not a class (old Boost; see SF #502085)
        is_metaclass = False

    if is_metaclass:
        return _class_reduce(obj)
    elif isinstance(obj, types.FunctionType):
        return _function_reduce(obj, pickler.globals_ref)
    else:
        # fallback to save_global
        return NotImplementedError


class CloudPickler(Pickler):
    """Fast C Pickler extension with additional reducing routines

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
        self.global_hook = _reduce_global
        self.proto = int(protocol)

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