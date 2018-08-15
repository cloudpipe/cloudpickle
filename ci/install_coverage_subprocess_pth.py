# Make it possible to enable test coverage reporting for Python
# code run in children processes.
# http://coverage.readthedocs.io/en/latest/subprocess.html

import os.path as op
from distutils.sysconfig import get_python_lib

FILE_CONTENT = u"""\
import coverage; coverage.process_startup()
"""

filename = op.join(get_python_lib(), 'coverage_subprocess.pth')
with open(filename, 'wb') as f:
    f.write(FILE_CONTENT.encode('ascii'))

print('Installed subprocess coverage support: %s' % filename)
