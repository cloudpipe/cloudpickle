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
