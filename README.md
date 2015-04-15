# cloudpickle
Cloudpickle as a standalone python library


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
