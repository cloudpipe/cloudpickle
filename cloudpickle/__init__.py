from __future__ import absolute_import


from cloudpickle.cloudpickle import *  # noqa
from cloudpickle.cloudpickle_fast import CloudPickler, dumps, dump  # noqa

Pickler = CloudPickler

__version__ = '1.5.1dev0'
