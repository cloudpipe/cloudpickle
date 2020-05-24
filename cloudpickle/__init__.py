from __future__ import absolute_import

import sys
import pickle
import types
import warnings

import cloudpickle.cloudpickle as cp


class CustomModuleType(types.ModuleType):
    def __getattr__(self, name):
        if name == 'Pickler':
            warnings.warn(
                'Pickler will point to Cloudpickler in two releases.',
                FutureWarning
            )
            return self._Pickler
        raise AttributeError

    def __reduce__(self):
        return __import__, ("cloudpickle.cloudpickle",)


cp.__class__ = CustomModuleType

if sys.version_info[:2] >= (3, 7):
    def __getattr__(name):
        return cp.__class__.__getattr__(cp, name)

from cloudpickle.cloudpickle import *

if sys.version_info[:2] >= (3, 8):
    from cloudpickle.cloudpickle_fast import CloudPickler, dumps, dump

__version__ = '1.5.0dev0'
