import abc
import pickle

import pytest

from .testutils import subprocess_pickle_echo


@pytest.mark.parametrize("descriptor_type", [property, abc.abstractproperty])
@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_property_descriptor_type(descriptor_type, protocol):
    def get_value(instance):
        return instance["value"]

    def set_value(instance, value):
        instance["value"] = value

    def delete_value(instance):
        del instance["value"]

    descriptor = descriptor_type(
        get_value, set_value, delete_value, "Property documentation."
    )
    clone = subprocess_pickle_echo(descriptor, protocol=protocol)

    assert type(clone) is descriptor_type
    assert clone.__isabstractmethod__ == descriptor.__isabstractmethod__
    assert clone.__doc__ == descriptor.__doc__
    values = {"value": 1}
    assert clone.__get__(values, dict) == 1
    clone.__set__(values, 2)
    assert clone.__get__(values, dict) == 2
    clone.__delete__(values)
    assert "value" not in values


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_abstract_property_blocks_incomplete_subclasses(legacy, protocol):
    def get_value(self):
        return "abstract placeholder"

    if legacy:
        descriptor = abc.abstractproperty(get_value)
    else:
        descriptor = property(abc.abstractmethod(get_value))

    class Base(abc.ABC):
        value = descriptor

    clone = subprocess_pickle_echo(Base, protocol=protocol)

    class Incomplete(clone):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()

    class Concrete(clone):
        @property
        def value(self):
            return 42

    assert Concrete().value == 42
    assert clone.__abstractmethods__ == frozenset({"value"})
