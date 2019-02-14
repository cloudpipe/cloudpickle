"""
New, fast version of the Cloudpickler.

This new Cloudpickler class can now extend the fast C Pickler instead of the
previous pythonic Pickler. Because this functionality is only available for
python versions 3.8+, a lot of backward-compatibilty code is also removed.
"""
from _pickle import Pickler
