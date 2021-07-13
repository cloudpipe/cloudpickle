"""
In the distributed computing setting, this file plays the role of a "local
development" file, e.g. a file that is importable locally, but unimportable in
remote workers. Constructs defined in this file and usually pickled by
reference should instead flagged to cloudpickle for pickling by value: this is
done using the register_pickle_by_value api exposed by cloudpickle.
"""
import typing


def local_function():
    return "hello from a function importable locally!"


class LocalClass:
    def method(self):
        return "hello from a class importable locally"


LocalT = typing.TypeVar("LocalT")
