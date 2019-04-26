from __future__ import absolute_import

import sys
import pickle


if sys.version_info[:2] >= (3, 8):
    from cloudpickle.cloudpickle_fast import *
else:
    from cloudpickle.cloudpickle import *

__version__ = '1.2.0.dev0'
