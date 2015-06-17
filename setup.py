#!/usr/bin/env python

import sys

from setuptools import setup
from setuptools import find_packages


def is_numpy_installed():
    try:
        import numpy
    except ImportError:
        return False
    return True


def setup_package():
    metadata = dict(
        name='sparkit-learn',
        version="0.2.4",
        description='Scikit-learn on PySpark',
        author='Krisztian Szucs, Andras Fulop',
        author_email='krisztian.szucs@lensa.com, andras.fulop@lensa.com',
        license='Apache License, Version 2.0',
        url='https://github.com/lensacom/sparkit-learn',
        packages=find_packages(),
        long_description=open('./README.rst').read(),
        install_requires=open('./requirements.txt').read().split()
    )
    if not (len(sys.argv) >= 2
            and ('--help' in sys.argv[1:] or sys.argv[1]
                 in ('--help-commands', 'egg_info', '--version', 'clean'))):
        if is_numpy_installed() is False:
            raise ImportError("Numerical Python (NumPy) is not installed.\n"
                              "sparkit-learn requires NumPy.\n"
                              "Installation instructions are available on scikit-learn website: "
                              "http://scikit-learn.org/stable/install.html\n")

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
