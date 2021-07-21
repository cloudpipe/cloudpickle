import sys
import types


# #354: To emulate package capabilities while being a single file, an extension
# module (for instance a mod.so file) can dynamically create a module object
# (most likely using the *package_name*.*parent_module_name*.*submodule_name*
# naming convention to be compatible with ``import`` semantics).  Internally,
# it will use the Python/C API ``PyImport_AddModule(submodule_qualified_name)``
# utility, which creates a module object and adds it inside ``sys.modules``. A
# module created this way IS a dynamic module. However, because the ``import``
# machinery automatically imports the parent package/module of a submodule
# before importing the submodule itself, and importing the parent
# package/module creates and append the submodule to sys.modules (sys.modules
# acts as a cache for the import machinery), this submodule, albeit dynamic, is
# importable. To detect this, we need to recursively check the parent modules
# of this submodule to see if the parent module is importable. If yes, we
# reasonably assume that the submodule named using the aforementioned
# hierarchised convention has been created during the import of its parent
# module.  The following lines emulate such a behavior without being a compiled
# extension module.

submodule_name = '_cloudpickle_testpkg.mod.dynamic_submodule'
dynamic_submodule = types.ModuleType(submodule_name)

# This line allows the dynamic_module to be imported using either one of:
# - ``from _cloudpickle_testpkg.mod import dynamic_submodule``
# - ``import _cloudpickle_testpkg.mod.dynamic_submodule``
sys.modules[submodule_name] = dynamic_submodule
# Both lines will make importlib try to get the module from sys.modules after
# importing the parent module, before trying getattr(mod, 'dynamic_submodule'),
# so this dynamic module could be binded to another name. This behavior is
# demonstrated with `dynamic_submodule_two`

submodule_name_two = '_cloudpickle_testpkg.mod.dynamic_submodule_two'
# Notice the inconsistent name binding, breaking attribute lookup-based import
# attempts.
another_submodule = types.ModuleType(submodule_name_two)
sys.modules[submodule_name_two] = another_submodule


# In this third case, the module is not added to sys.modules, and can only be
# imported using attribute lookup-based imports.
submodule_three = types.ModuleType(
    '_cloudpickle_testpkg.mod.dynamic_submodule_three'
)
code = """
def f(x):
    return x
"""

exec(code, vars(submodule_three))

# What about a dynamic submodule inside a dynamic submodule inside an
# importable module?
subsubmodule_name = (
    '_cloudpickle_testpkg.mod.dynamic_submodule.dynamic_subsubmodule'
)
dynamic_subsubmodule = types.ModuleType(subsubmodule_name)
dynamic_submodule.dynamic_subsubmodule = dynamic_subsubmodule
sys.modules[subsubmodule_name] = dynamic_subsubmodule


def module_function():
    return "hello from a module!"


global_variable = "some global variable"


def module_function_with_global():
    global global_variable
    return global_variable
