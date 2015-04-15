# cloudpickle

[![Build Status](https://travis-ci.org/cloudpipe/cloudpickle.svg?branch=master
    )](https://travis-ci.org/cloudpipe/cloudpickle)

`cloudpickle` makes it possible to serialize Python constructs not supported
by the default `pickle` module from the Python standard library.

`cloudpickle` is especially useful for cluster computing where Python
expressions are shipped over the network to execute on remote hosts, possibly
close to the data.

Among other things, `cloudpickle` supports pickling for lambda expressions,
functions and classes defined interactively in the `__main__` module.

Example:

    >>> import cloudpickle
    >>> squared = lambda x: x ** 2
    >>> pickled_lambda = cloudpickle.dumps(squared)

    >>> import pickle
    >>> new_squared = pickle.loads(pickled_lambda)
    >>> new_squared(2)
    4


Running the tests
-----------------

- With `tox`, to test run the tests for all the supported versions of
  Python and PyPy:

      pip install tox
      tox

  or alternatively for a specific environment:

      tox -e py27


- With `py.test` to only run the tests for your current version of
  Python:

      pip install -r dev-requirements.txt
      PYTHONPATH='.:tests' py.test
