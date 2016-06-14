#!/usr/bin/env python

import sys

from setuptools import find_packages, setup


def setup_package():
    metadata = dict(
        name='sparkit-learn',
        version='0.2.6',
        description='Scikit-learn on PySpark',
        author='Krisztian Szucs, Andras Fulop',
        author_email='krisztian.szucs@lensa.com, andras.fulop@lensa.com',
        license='Apache License, Version 2.0',
        url='https://github.com/lensacom/sparkit-learn',
        packages=find_packages(),
        long_description=open('./README.rst').read(),
        install_requires=open('./requirements.txt').read().split()
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
