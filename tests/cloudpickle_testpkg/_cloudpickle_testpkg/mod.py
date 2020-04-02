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
sys.modules[submodule_name] = dynamic_submodule


def module_function():
    return "hello from a module!"
