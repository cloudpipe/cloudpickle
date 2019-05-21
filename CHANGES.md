1.2.0
=====

- Support pickling of classmethod and staticmethod objects in python2.
  arguments. ([issue #262](https://github.com/cloudpipe/cloudpickle/pull/262))

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
