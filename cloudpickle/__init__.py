from __future__ import absolute_import

from cloudpickle.compat import pickle


from cloudpickle.cloudpickle import *
if pickle.HIGHEST_PROTOCOL >= 5:
    from cloudpickle.cloudpickle_fast import CloudPickler, dumps, dump

__version__ = '1.5.0dev0'
