#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requirements = [
]

test_requirements = [
    'pytest',
    'pytest-cov'
]

dist = setup(
    name='cloudpickle',
    version='0.1.0',
    description='Extended pickling support for Python objects',
    author='Cloudpipe',
    author_email='cloudpipe@googlegroups.com',
    url='https://github.com/cloudpipe/cloudpickle',
    install_requires=requirements,
    license='LICENSE.txt',
    packages=['cloudpickle'],
    long_description=open('README.md').read(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: System :: Distributed Computing',
        ],
    test_suite='tests',
    tests_require=test_requirements
)
