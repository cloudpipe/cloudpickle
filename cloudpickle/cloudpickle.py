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

import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing

from .compat import pickle
from collections import OrderedDict
# The following import is required to be imported in the cloudpickle
# namespace to be able to load pickle files generated with older versions of
# cloudpickle. See: tests/test_backward_compat.py
from types import CellType  # noqa: F401
from pickle import _getattribute


# cloudpickle is meant for inter process communication: we expect all
# communicating processes to run the same Python version hence we favor
# communication speed over compatibility:
DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL

# Names of modules whose resources should be treated as dynamic.
_PICKLE_BY_VALUE_MODULES = set()

# Track the provenance of reconstructed dynamic classes to make it possible to
# reconstruct instances from the matching singleton class definition when
# appropriate and preserve the usual "isinstance" semantics of Python objects.
_DYNAMIC_CLASS_TRACKER_BY_CLASS = weakref.WeakKeyDictionary()
_DYNAMIC_CLASS_TRACKER_BY_ID = weakref.WeakValueDictionary()
_DYNAMIC_CLASS_TRACKER_LOCK = threading.Lock()

PYPY = platform.python_implementation() == "PyPy"

builtin_code_type = None
if PYPY:
    # builtin-code objects only exist in pypy
    builtin_code_type = type(float.__new__.__code__)

_extract_code_globals_cache = weakref.WeakKeyDictionary()


def _get_or_create_tracker_id(class_def):
    with _DYNAMIC_CLASS_TRACKER_LOCK:
        class_tracker_id = _DYNAMIC_CLASS_TRACKER_BY_CLASS.get(class_def)
        if class_tracker_id is None:
            class_tracker_id = uuid.uuid4().hex
            _DYNAMIC_CLASS_TRACKER_BY_CLASS[class_def] = class_tracker_id
            _DYNAMIC_CLASS_TRACKER_BY_ID[class_tracker_id] = class_def
    return class_tracker_id


def _lookup_class_or_track(class_tracker_id, class_def):
    if class_tracker_id is not None:
        with _DYNAMIC_CLASS_TRACKER_LOCK:
            class_def = _DYNAMIC_CLASS_TRACKER_BY_ID.setdefault(
                class_tracker_id, class_def)
            _DYNAMIC_CLASS_TRACKER_BY_CLASS[class_def] = class_tracker_id
    return class_def


def register_pickle_by_value(module):
    """Register a module to make it functions and classes picklable by value.

    By default, functions and classes that are attributes of an importable
    module are to be pickled by reference, that is relying on re-importing
    the attribute from the module at load time.

    If `register_pickle_by_value(module)` is called, all its functions and
    classes are subsequently to be pickled by value, meaning that they can
    be loaded in Python processes where the module is not importable.

    This is especially useful when developing a module in a distributed
    execution environment: restarting the client Python process with the new
    source code is enough: there is no need to re-install the new version
    of the module on all the worker nodes nor to restart the workers.

    Note: this feature is considered experimental. See the cloudpickle
    README.md file for more details and limitations.
    """
    if not isinstance(module, types.ModuleType):
        raise ValueError(
            f"Input should be a module object, got {str(module)} instead"
        )
    # In the future, cloudpickle may need a way to access any module registered
    # for pickling by value in order to introspect relative imports inside
    # functions pickled by value. (see
    # https://github.com/cloudpipe/cloudpickle/pull/417#issuecomment-873684633).
    # This access can be ensured by checking that module is present in
    # sys.modules at registering time and assuming that it will still be in
    # there when accessed during pickling. Another alternative would be to
    # store a weakref to the module. Even though cloudpickle does not implement
    # this introspection yet, in order to avoid a possible breaking change
    # later, we still enforce the presence of module inside sys.modules.
    if module.__name__ not in sys.modules:
        raise ValueError(
            f"{module} was not imported correctly, have you used an "
            f"`import` statement to access it?"
        )
    _PICKLE_BY_VALUE_MODULES.add(module.__name__)


def unregister_pickle_by_value(module):
    """Unregister that the input module should be pickled by value."""
    if not isinstance(module, types.ModuleType):
        raise ValueError(
            f"Input should be a module object, got {str(module)} instead"
        )
    if module.__name__ not in _PICKLE_BY_VALUE_MODULES:
        raise ValueError(f"{module} is not registered for pickle by value")
    else:
        _PICKLE_BY_VALUE_MODULES.remove(module.__name__)


def list_registry_pickle_by_value():
    return _PICKLE_BY_VALUE_MODULES.copy()


def _is_registered_pickle_by_value(module):
    module_name = module.__name__
    if module_name in _PICKLE_BY_VALUE_MODULES:
        return True
    while True:
        parent_name = module_name.rsplit(".", 1)[0]
        if parent_name == module_name:
            break
        if parent_name in _PICKLE_BY_VALUE_MODULES:
            return True
        module_name = parent_name
    return False


def _whichmodule(obj, name):
    """Find the module an object belongs to.

    This function differs from ``pickle.whichmodule`` in two ways:
    - it does not mangle the cases where obj's module is __main__ and obj was
      not found in any module.
    - Errors arising during module introspection are ignored, as those errors
      are considered unwanted side effects.
    """
    module_name = getattr(obj, '__module__', None)

    if module_name is not None:
        return module_name
    # Protect the iteration by using a copy of sys.modules against dynamic
    # modules that trigger imports of other modules upon calls to getattr or
    # other threads importing at the same time.
    for module_name, module in sys.modules.copy().items():
        # Some modules such as coverage can inject non-module objects inside
        # sys.modules
        if (
                module_name == '__main__' or
                module is None or
                not isinstance(module, types.ModuleType)
        ):
            continue
        try:
            if _getattribute(module, name)[0] is obj:
                return module_name
        except Exception:
            pass
    return None


def _should_pickle_by_reference(obj, name=None):
    """Test whether an function or a class should be pickled by reference

     Pickling by reference means by that the object (typically a function or a
     class) is an attribute of a module that is assumed to be importable in the
     target Python environment. Loading will therefore rely on importing the
     module and then calling `getattr` on it to access the function or class.

     Pickling by reference is the only option to pickle functions and classes
     in the standard library. In cloudpickle the alternative option is to
     pickle by value (for instance for interactively or locally defined
     functions and classes or for attributes of modules that have been
     explicitly registered to be pickled by value.
     """
    if isinstance(obj, types.FunctionType) or issubclass(type(obj), type):
        module_and_name = _lookup_module_and_qualname(obj, name=name)
        if module_and_name is None:
            return False
        module, name = module_and_name
        return not _is_registered_pickle_by_value(module)

    elif isinstance(obj, types.ModuleType):
        # We assume that sys.modules is primarily used as a cache mechanism for
        # the Python import machinery. Checking if a module has been added in
        # is sys.modules therefore a cheap and simple heuristic to tell us
        # whether we can assume that a given module could be imported by name
        # in another Python process.
        if _is_registered_pickle_by_value(obj):
            return False
        return obj.__name__ in sys.modules
    else:
        raise TypeError(
            "cannot check importability of {} instances".format(
                type(obj).__name__)
        )


def _lookup_module_and_qualname(obj, name=None):
    if name is None:
        name = getattr(obj, '__qualname__', None)
    if name is None:  # pragma: no cover
        # This used to be needed for Python 2.7 support but is probably not
        # needed anymore. However we keep the __name__ introspection in case
        # users of cloudpickle rely on this old behavior for unknown reasons.
        name = getattr(obj, '__name__', None)

    module_name = _whichmodule(obj, name)

    if module_name is None:
        # In this case, obj.__module__ is None AND obj was not found in any
        # imported module. obj is thus treated as dynamic.
        return None

    if module_name == "__main__":
        return None

    # Note: if module_name is in sys.modules, the corresponding module is
    # assumed importable at unpickling time. See #357
    module = sys.modules.get(module_name, None)
    if module is None:
        # The main reason why obj's module would not be imported is that this
        # module has been dynamically created, using for example
        # types.ModuleType. The other possibility is that module was removed
        # from sys.modules after obj was created/imported. But this case is not
        # supported, as the standard pickle does not support it either.
        return None

    try:
        obj2, parent = _getattribute(module, name)
    except AttributeError:
        # obj was not found inside the module it points to
        return None
    if obj2 is not obj:
        return None
    return module, name


def _extract_code_globals(co):
    """
    Find all globals names read or written to by codeblock co
    """
    out_names = _extract_code_globals_cache.get(co)
    if out_names is None:
        # We use a dict with None values instead of a set to get a
        # deterministic order and avoid introducing non-deterministic pickle
        # bytes as a results.
        out_names = {name: None for name in _walk_global_ops(co)}

        # Declaring a function inside another one using the "def ..." syntax
        # generates a constant code object corresponding to the one of the
        # nested function's As the nested function may itself need global
        # variables, we need to introspect its code, extract its globals, (look
        # for code object in it's co_consts attribute..) and add the result to
        # code_globals
        if co.co_consts:
            for const in co.co_consts:
                if isinstance(const, types.CodeType):
                    out_names.update(_extract_code_globals(const))

        _extract_code_globals_cache[co] = out_names

    return out_names


def _find_imported_submodules(code, top_level_dependencies):
    """
    Find currently imported submodules used by a function.

    Submodules used by a function need to be detected and referenced for the
    function to work correctly at depickling time. Because submodules can be
    referenced as attribute of their parent package (``package.submodule``), we
    need a special introspection technique that does not rely on GLOBAL-related
    opcodes to find references of them in a code object.

    Example:
    ```
    import concurrent.futures
    import cloudpickle
    def func():
        x = concurrent.futures.ThreadPoolExecutor
    if __name__ == '__main__':
        cloudpickle.dumps(func)
    ```
    The globals extracted by cloudpickle in the function's state include the
    concurrent package, but not its submodule (here, concurrent.futures), which
    is the module used by func. Find_imported_submodules will detect the usage
    of concurrent.futures. Saving this module alongside with func will ensure
    that calling func once depickled does not fail due to concurrent.futures
    not being imported
    """

    subimports = []
    # check if any known dependency is an imported package
    for x in top_level_dependencies:
        if (isinstance(x, types.ModuleType) and
                hasattr(x, '__package__') and x.__package__):
            # check if the package has any currently loaded sub-imports
            prefix = x.__name__ + '.'
            # A concurrent thread could mutate sys.modules,
            # make sure we iterate over a copy to avoid exceptions
            for name in list(sys.modules):
                # Older versions of pytest will add a "None" module to
                # sys.modules.
                if name is not None and name.startswith(prefix):
                    # check whether the function can address the sub-module
                    tokens = set(name[len(prefix):].split('.'))
                    if not tokens - set(code.co_names):
                        subimports.append(sys.modules[name])
    return subimports


# relevant opcodes
STORE_GLOBAL = opcode.opmap['STORE_GLOBAL']
DELETE_GLOBAL = opcode.opmap['DELETE_GLOBAL']
LOAD_GLOBAL = opcode.opmap['LOAD_GLOBAL']
GLOBAL_OPS = (STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL)
HAVE_ARGUMENT = dis.HAVE_ARGUMENT
EXTENDED_ARG = dis.EXTENDED_ARG


_BUILTIN_TYPE_NAMES = {}
for k, v in types.__dict__.items():
    if type(v) is type:
        _BUILTIN_TYPE_NAMES[v] = k


def _builtin_type(name):
    if name == "ClassType":  # pragma: no cover
        # Backward compat to load pickle files generated with cloudpickle
        # < 1.3 even if loading pickle files from older versions is not
        # officially supported.
        return type
    return getattr(types, name)


def _walk_global_ops(code):
    """
    Yield referenced name for all global-referencing instructions in *code*.
    """
    for instr in dis.get_instructions(code):
        op = instr.opcode
        if op in GLOBAL_OPS:
            yield instr.argval


def _extract_class_dict(cls):
    """Retrieve a copy of the dict of a class without the inherited methods"""
    clsdict = dict(cls.__dict__)  # copy dict proxy to a dict
    if len(cls.__bases__) == 1:
        inherited_dict = cls.__bases__[0].__dict__
    else:
        inherited_dict = {}
        for base in reversed(cls.__bases__):
            inherited_dict.update(base.__dict__)
    to_remove = []
    for name, value in clsdict.items():
        try:
            base_value = inherited_dict[name]
            if value is base_value:
                to_remove.append(name)
        except KeyError:
            pass
    for name in to_remove:
        clsdict.pop(name)
    return clsdict


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


# including pickles unloading functions in this namespace
load = pickle.load
loads = pickle.loads


def subimport(name):
    # We cannot do simply: `return __import__(name)`: Indeed, if ``name`` is
    # the name of a submodule, __import__ will return the top-level root module
    # of this submodule. For instance, __import__('os.path') returns the `os`
    # module.
    __import__(name)
    return sys.modules[name]


def dynamic_subimport(name, vars):
    mod = types.ModuleType(name)
    mod.__dict__.update(vars)
    mod.__dict__['__builtins__'] = builtins.__dict__
    return mod


def _get_cell_contents(cell):
    try:
        return cell.cell_contents
    except ValueError:
        # Handle empty cells explicitly with a sentinel value.
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
class _empty_cell_value:
    """sentinel for empty closures
    """
    @classmethod
    def __reduce__(cls):
        return cls.__name__


def _make_function(code, globals, name, argdefs, closure):
    # Setting __builtins__ in globals is needed for nogil CPython.
    globals["__builtins__"] = __builtins__
    return types.FunctionType(code, globals, name, argdefs, closure)


def _make_empty_cell():
    if False:
        # trick the compiler into creating an empty cell in our lambda
        cell = None
        raise AssertionError('this route should not be executed')

    return (lambda: cell).__closure__[0]


def _make_cell(value=_empty_cell_value):
    cell = _make_empty_cell()
    if value is not _empty_cell_value:
        cell.cell_contents = value
    return cell


def _make_skeleton_class(type_constructor, name, bases, type_kwargs,
                         class_tracker_id, extra):
    """Build dynamic class with an empty __dict__ to be filled once memoized

    If class_tracker_id is not None, try to lookup an existing class definition
    matching that id. If none is found, track a newly reconstructed class
    definition under that id so that other instances stemming from the same
    class id will also reuse this class definition.

    The "extra" variable is meant to be a dict (or None) that can be used for
    forward compatibility shall the need arise.
    """
    skeleton_class = types.new_class(
        name, bases, {'metaclass': type_constructor},
        lambda ns: ns.update(type_kwargs)
    )
    return _lookup_class_or_track(class_tracker_id, skeleton_class)


def _rehydrate_skeleton_class(skeleton_class, class_dict):
    """Put attributes from `class_dict` back on `skeleton_class`.

    See CloudPickler.save_dynamic_class for more info.
    """
    registry = None
    for attrname, attr in class_dict.items():
        if attrname == "_abc_impl":
            registry = attr
        else:
            setattr(skeleton_class, attrname, attr)
    if registry is not None:
        for subclass in registry:
            skeleton_class.register(subclass)

    return skeleton_class


def _make_skeleton_enum(bases, name, qualname, members, module,
                        class_tracker_id, extra):
    """Build dynamic enum with an empty __dict__ to be filled once memoized

    The creation of the enum class is inspired by the code of
    EnumMeta._create_.

    If class_tracker_id is not None, try to lookup an existing enum definition
    matching that id. If none is found, track a newly reconstructed enum
    definition under that id so that other instances stemming from the same
    class id will also reuse this enum definition.

    The "extra" variable is meant to be a dict (or None) that can be used for
    forward compatibility shall the need arise.
    """
    # enums always inherit from their base Enum class at the last position in
    # the list of base classes:
    enum_base = bases[-1]
    metacls = enum_base.__class__
    classdict = metacls.__prepare__(name, bases)

    for member_name, member_value in members.items():
        classdict[member_name] = member_value
    enum_class = metacls.__new__(metacls, name, bases, classdict)
    enum_class.__module__ = module
    enum_class.__qualname__ = qualname

    return _lookup_class_or_track(class_tracker_id, enum_class)


def _make_typevar(name, bound, constraints, covariant, contravariant,
                  class_tracker_id):
    tv = typing.TypeVar(
        name, *constraints, bound=bound,
        covariant=covariant, contravariant=contravariant
    )
    if class_tracker_id is not None:
        return _lookup_class_or_track(class_tracker_id, tv)
    else:  # pragma: nocover
        # Only for Python 3.5.3 compat.
        return tv


def _decompose_typevar(obj):
    return (
        obj.__name__, obj.__bound__, obj.__constraints__,
        obj.__covariant__, obj.__contravariant__,
        _get_or_create_tracker_id(obj),
    )


def _typevar_reduce(obj):
    # TypeVar instances require the module information hence why we
    # are not using the _should_pickle_by_reference directly
    module_and_name = _lookup_module_and_qualname(obj, name=obj.__name__)

    if module_and_name is None:
        return (_make_typevar, _decompose_typevar(obj))
    elif _is_registered_pickle_by_value(module_and_name[0]):
        return (_make_typevar, _decompose_typevar(obj))

    return (getattr, module_and_name)


def _get_bases(typ):
    if '__orig_bases__' in getattr(typ, '__dict__', {}):
        # For generic types (see PEP 560)
        # Note that simply checking `hasattr(typ, '__orig_bases__')` is not
        # correct.  Subclasses of a fully-parameterized generic class does not
        # have `__orig_bases__` defined, but `hasattr(typ, '__orig_bases__')`
        # will return True because it's defined in the base class.
        bases_attr = '__orig_bases__'
    else:
        # For regular class objects
        bases_attr = '__bases__'
    return getattr(typ, bases_attr)


def _make_dict_keys(obj, is_ordered=False):
    if is_ordered:
        return OrderedDict.fromkeys(obj).keys()
    else:
        return dict.fromkeys(obj).keys()


def _make_dict_values(obj, is_ordered=False):
    if is_ordered:
        return OrderedDict((i, _) for i, _ in enumerate(obj)).values()
    else:
        return {i: _ for i, _ in enumerate(obj)}.values()


def _make_dict_items(obj, is_ordered=False):
    if is_ordered:
        return OrderedDict(obj).items()
    else:
        return obj.items()
