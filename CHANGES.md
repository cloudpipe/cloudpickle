3.1.0
=====

- Some improvements to make cloudpickle more deterministic when pickling
  dynamic functions and classes, in particular with CPython 3.13.
  ([PR #524](https://github.com/cloudpipe/cloudpickle/pull/524) and
   [PR #534](https://github.com/cloudpipe/cloudpickle/pull/534))

- Fix a problem with the joint usage of cloudpickle's `_whichmodule` and
  `multiprocessing`.
  ([PR #529](https://github.com/cloudpipe/cloudpickle/pull/529))

3.0.0
=====

- Officially support Python 3.12 and drop support for Python 3.6 and 3.7.
  Dropping support for older Python versions made it possible to simplify the
  code base significantly, hopefully making it easier to contribute to and
  maintain the project.
  ([PR #517](https://github.com/cloudpipe/cloudpickle/pull/517))

- Fix pickling of dataclasses and their instances.
  ([issue #386](https://github.com/cloudpipe/cloudpickle/issues/386),
   [PR #513](https://github.com/cloudpipe/cloudpickle/pull/513))

- Any color you like as long as it's black.
  ([PR #521](https://github.com/cloudpipe/cloudpickle/pull/521))

- Drop `setup.py` and `setuptools` in favor of `pyproject.toml` and `flit`.
  ([PR #521](https://github.com/cloudpipe/cloudpickle/pull/521))

2.2.1
=====

- Fix pickling of NamedTuple in Python 3.9+.
  ([issue #460](https://github.com/cloudpipe/cloudpickle/issues/460))

2.2.0
=====

- Fix support of PyPy 3.8 and later.
  ([issue #455](https://github.com/cloudpipe/cloudpickle/issues/455))

2.1.0
=====

- Support for pickling `abc.abstractproperty`, `abc.abstractclassmethod`,
  and `abc.abstractstaticmethod`.
  ([PR #450](https://github.com/cloudpipe/cloudpickle/pull/450))

- Support for pickling subclasses of generic classes.
  ([PR #448](https://github.com/cloudpipe/cloudpickle/pull/448))

- Support and CI configuration for Python 3.11.
  ([PR #467](https://github.com/cloudpipe/cloudpickle/pull/467))

- Support for the experimental `nogil` variant of CPython
  ([PR #470](https://github.com/cloudpipe/cloudpickle/pull/470))

2.0.0
=====

- Python 3.5 is no longer supported.

- Support for registering modules to be serialised by value. This allows code
  defined in local modules to be serialised and executed remotely without those
  local modules installed on the remote machine.
  ([PR #417](https://github.com/cloudpipe/cloudpickle/pull/417))

- Fix a side effect altering dynamic modules at pickling time.
  ([PR #426](https://github.com/cloudpipe/cloudpickle/pull/426))

- Support for pickling type annotations on Python 3.10 as per [PEP 563](
  https://www.python.org/dev/peps/pep-0563/)
  ([PR #400](https://github.com/cloudpipe/cloudpickle/pull/400))

- Stricter parametrized type detection heuristics in
  _is_parametrized_type_hint to limit false positives.
  ([PR #409](https://github.com/cloudpipe/cloudpickle/pull/409))

- Support pickling / depickling of OrderedDict KeysView, ValuesView, and
  ItemsView, following similar strategy for vanilla Python dictionaries.
  ([PR #423](https://github.com/cloudpipe/cloudpickle/pull/423))

- Suppressed a source of non-determinism when pickling dynamically defined
  functions and handles the deprecation of co_lnotab in Python 3.10+.
  ([PR #428](https://github.com/cloudpipe/cloudpickle/pull/428))

1.6.0
=====

- `cloudpickle`'s pickle.Pickler subclass (currently defined as
  `cloudpickle.cloudpickle_fast.CloudPickler`) can and should now be accessed
  as `cloudpickle.Pickler`. This is the only officially supported way of
  accessing it.
  ([issue #366](https://github.com/cloudpipe/cloudpickle/issues/366))

- `cloudpickle` now supports pickling `dict_keys`, `dict_items` and
  `dict_values`.
  ([PR #384](https://github.com/cloudpipe/cloudpickle/pull/384))


1.5.0
=====

- Fix a bug causing cloudpickle to crash when pickling dynamically created,
  importable modules.
  ([issue #360](https://github.com/cloudpipe/cloudpickle/issues/354))

- Add optional dependency on `pickle5` to get improved performance on
  Python 3.6 and 3.7.
  ([PR #370](https://github.com/cloudpipe/cloudpickle/pull/370))

- Internal refactoring to ease the use of `pickle5` in cloudpickle
  for Python 3.6 and 3.7.
  ([PR #368](https://github.com/cloudpipe/cloudpickle/pull/368))


1.4.1
=====

- Fix incompatibilities between cloudpickle 1.4.0 and Python 3.5.0/1/2
  introduced by the new support of cloudpickle for pickling typing constructs.
  ([issue #360](https://github.com/cloudpipe/cloudpickle/issues/360))

- Restore compat with loading dynamic classes pickled with cloudpickle
  version 1.2.1 that would reference the `types.ClassType` attribute.
  ([PR #359](https://github.com/cloudpipe/cloudpickle/pull/359))


1.4.0
=====

**This version requires Python 3.5 or later**

- cloudpickle can now all pickle all constructs from the ``typing`` module
  and the ``typing_extensions`` library in Python 3.5+
  ([PR #318](https://github.com/cloudpipe/cloudpickle/pull/318))

- Stop pickling the annotations of a dynamic class for Python < 3.6
  (follow up on #276)
  ([issue #347](https://github.com/cloudpipe/cloudpickle/issues/347))

- Fix a bug affecting the pickling of dynamic `TypeVar` instances on Python 3.7+,
  and expand the support for pickling `TypeVar` instances (dynamic or non-dynamic)
  to Python 3.5-3.6 ([PR #350](https://github.com/cloudpipe/cloudpickle/pull/350))

- Add support for pickling dynamic classes subclassing `typing.Generic`
  instances on Python 3.7+
  ([PR #351](https://github.com/cloudpipe/cloudpickle/pull/351))

1.3.0
=====

- Fix a bug affecting dynamic modules occuring with modified builtins
  ([issue #316](https://github.com/cloudpipe/cloudpickle/issues/316))

- Fix a bug affecting cloudpickle when non-modules objects are added into
  sys.modules
  ([PR #326](https://github.com/cloudpipe/cloudpickle/pull/326)).

- Fix a regression in cloudpickle and python3.8 causing an error when trying to
  pickle property objects.
  ([PR #329](https://github.com/cloudpipe/cloudpickle/pull/329)).

- Fix a bug when a thread imports a module while cloudpickle iterates
  over the module list
  ([PR #322](https://github.com/cloudpipe/cloudpickle/pull/322)).

- Add support for out-of-band pickling (Python 3.8 and later).
  https://docs.python.org/3/library/pickle.html#example
  ([issue #308](https://github.com/cloudpipe/cloudpickle/pull/308))

- Fix a side effect that would redefine `types.ClassTypes` as `type`
  when importing cloudpickle.
  ([issue #337](https://github.com/cloudpipe/cloudpickle/pull/337))

- Fix a bug affecting subclasses of slotted classes.
  ([issue #311](https://github.com/cloudpipe/cloudpickle/issues/311))

- Dont pickle the abc cache of dynamically defined classes for Python 3.6-
  (This was already the case for python3.7+)
  ([issue #302](https://github.com/cloudpipe/cloudpickle/issues/302))

1.2.2
=====

- Revert the change introduced in
  ([issue #276](https://github.com/cloudpipe/cloudpickle/pull/276))
  attempting to pickle functions annotations for Python 3.4 to 3.6. It is not
  possible to pickle complex typing constructs for those versions (see
  [issue #193]( https://github.com/cloudpipe/cloudpickle/issues/193))

- Fix a bug affecting bound classmethod saving on Python 2.
  ([issue #288](https://github.com/cloudpipe/cloudpickle/issues/288))

- Add support for pickling "getset" descriptors
  ([issue #290](https://github.com/cloudpipe/cloudpickle/pull/290))

1.2.1
=====

- Restore (partial) support for Python 3.4 for downstream projects that have
  LTS versions that would benefit from cloudpickle bug fixes.


1.2.0
=====

- Leverage the C-accelerated Pickler new subclassing API (available in Python
  3.8) in cloudpickle. This allows cloudpickle to pickle Python objects up to
  30 times faster.
  ([issue #253](https://github.com/cloudpipe/cloudpickle/pull/253))

- Support pickling of classmethod and staticmethod objects in python2.
  arguments. ([issue #262](https://github.com/cloudpipe/cloudpickle/pull/262))

- Add support to pickle type annotations for Python 3.5 and 3.6 (pickling type
  annotations was already supported for Python 3.7, Python 3.4 might also work
  but is no longer officially supported by cloudpickle)
  ([issue #276](https://github.com/cloudpipe/cloudpickle/pull/276))

- Internal refactoring to proactively detect dynamic functions and classes when
  pickling them.  This refactoring also yields small performance improvements
  when pickling dynamic classes (~10%)
  ([issue #273](https://github.com/cloudpipe/cloudpickle/pull/273))

1.1.1
=====

- Minor release to fix a packaging issue (Markdown formatting of the long
  description rendered on pypi.org). The code itself is the same as 1.1.0.

1.1.0
=====

- Support the pickling of interactively-defined functions with positional-only
  arguments. ([issue #266](https://github.com/cloudpipe/cloudpickle/pull/266))

- Track the provenance of dynamic classes and enums so as to preseve the
  usual `isinstance` relationship between pickled objects and their
  original class defintions.
  ([issue #246](https://github.com/cloudpipe/cloudpickle/pull/246))

1.0.0
=====

- Fix a bug making functions with keyword-only arguments forget the default
  values of these arguments after being pickled.
  ([issue #264](https://github.com/cloudpipe/cloudpickle/pull/264))

0.8.1
=====

- Fix a bug (already present before 0.5.3 and re-introduced in 0.8.0)
  affecting relative import instructions inside depickled functions
  ([issue #254](https://github.com/cloudpipe/cloudpickle/pull/254))

0.8.0
=====

- Add support for pickling interactively defined dataclasses.
  ([issue #245](https://github.com/cloudpipe/cloudpickle/pull/245))

- Global variables referenced by functions pickled by cloudpickle are now
  unpickled in a new and isolated namespace scoped by the CloudPickler
  instance. This restores the (previously untested) behavior of cloudpickle
  prior to changes done in 0.5.4 for functions defined in the `__main__`
  module, and 0.6.0/1 for other dynamic functions.

0.7.0
=====

- Correctly serialize dynamically defined classes that have a `__slots__`
  attribute.
  ([issue #225](https://github.com/cloudpipe/cloudpickle/issues/225))


0.6.1
=====

- Fix regression in 0.6.0 which breaks the pickling of local function defined
  in a module, making it impossible to access builtins.
  ([issue #211](https://github.com/cloudpipe/cloudpickle/issues/211))


0.6.0
=====

- Ensure that unpickling a function defined in a dynamic module several times
  sequentially does not reset the values of global variables.
  ([issue #187](https://github.com/cloudpipe/cloudpickle/issues/205))

- Restrict the ability to pickle annotations to python3.7+ ([issue #193](
  https://github.com/cloudpipe/cloudpickle/issues/193) and [issue #196](
  https://github.com/cloudpipe/cloudpickle/issues/196))

- Stop using the deprecated `imp` module under Python 3.
  ([issue #207](https://github.com/cloudpipe/cloudpickle/issues/207))

- Fixed pickling issue with singleton types `NoneType`, `type(...)` and
  `type(NotImplemented)` ([issue #209](https://github.com/cloudpipe/cloudpickle/issues/209))


0.5.6
=====

- Ensure that unpickling a locally defined function that accesses the global
  variables of a module does not reset the values of the global variables if
  they are already initialized.
  ([issue #187](https://github.com/cloudpipe/cloudpickle/issues/187))


0.5.5
=====

- Fixed inconsistent version in `cloudpickle.__version__`.


0.5.4
=====

- Fixed a pickling issue for ABC in python3.7+ ([issue #180](
  https://github.com/cloudpipe/cloudpickle/issues/180)).

- Fixed a bug when pickling functions in `__main__` that access global
  variables ([issue #187](
  https://github.com/cloudpipe/cloudpickle/issues/187)).

0.5.3
=====
- Fixed a crash in Python 2 when serializing non-hashable instancemethods of built-in
  types ([issue #144](https://github.com/cloudpipe/cloudpickle/issues/144)).

- itertools objects can also pickled
  ([PR #156](https://github.com/cloudpipe/cloudpickle/pull/156)).

- `logging.RootLogger` can be also pickled
  ([PR #160](https://github.com/cloudpipe/cloudpickle/pull/160)).

0.5.2
=====

- Fixed a regression: `AttributeError` when loading pickles that hold a
  reference to a dynamically defined class from the `__main__` module.
  ([issue #131]( https://github.com/cloudpipe/cloudpickle/issues/131)).

- Make it possible to pickle classes and functions defined in faulty
  modules that raise an exception when trying to look-up their attributes
  by name.


0.5.1
=====

- Fixed `cloudpickle.__version__`.

0.5.0
=====

- Use `pickle.HIGHEST_PROTOCOL` by default.

0.4.4
=====

- `logging.RootLogger` can be also pickled
  ([PR #160](https://github.com/cloudpipe/cloudpickle/pull/160)).

0.4.3
=====

- Fixed a regression: `AttributeError` when loading pickles that hold a
  reference to a dynamically defined class from the `__main__` module.
  ([issue #131]( https://github.com/cloudpipe/cloudpickle/issues/131)).

- Fixed a crash in Python 2 when serializing non-hashable instancemethods of built-in
  types. ([issue #144](https://github.com/cloudpipe/cloudpickle/issues/144))

0.4.2
=====

- Restored compatibility with pickles from 0.4.0.
- Handle the `func.__qualname__` attribute.

0.4.1
=====

- Fixed a crash when pickling dynamic classes whose `__dict__` attribute was
  defined as a [`property`](https://docs.python.org/3/library/functions.html#property).
  Most notably, this affected dynamic [namedtuples](https://docs.python.org/2/library/collections.html#namedtuple-factory-function-for-tuples-with-named-fields)
  in Python 2. (https://github.com/cloudpipe/cloudpickle/pull/113)
- Cloudpickle now preserves the `__module__` attribute of functions (https://github.com/cloudpipe/cloudpickle/pull/118/).
- Fixed a crash when pickling modules that don't have a `__package__` attribute (https://github.com/cloudpipe/cloudpickle/pull/116).

0.4.0
=====

* Fix functions with empty cells
* Allow pickling Logger objects
* Fix crash when pickling dynamic class cycles
* Ignore "None" mdoules added to sys.modules
* Support WeakSets and ABCMeta instances
* Remove non-standard `__transient__` support
* Catch exception from `pickle.whichmodule()`


0.3.1
=====

* Fix version information and ship a changelog

 0.3.0
=====

* Import submodules accessed by pickled functions
* Support recursive functions inside closures
* Fix `ResourceWarnings` and `DeprecationWarnings`
* Assume modules with `__file__` attribute are not dynamic

0.2.2
=====

* Support Python 3.6
* Support Tornado Coroutines
* Support builtin methods

0.2.1
=====

* Packaging fix

0.2.0
=====

* Support `method_descriptor`
* Support unbound instancemethods
* Fixes for PyPy3
* More support for pickling dynamic imports

0.1.0
=====

Released on 2015-04-16 from the (real) clouds somewhere between Montréal and
Atlanta.
