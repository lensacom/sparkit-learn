#!/usr/bin/env python

from setuptools import setup

import splearn

setup(
    name='sparkit-learn',
    version=str(splearn.__version__),
    description='Scikit-learn on PySpark',
    author='Krisztian Szucs, Andras Fulop',
    author_email='krisztian.szucs@lensa.com, andras.fulop@lensa.com',
    url='https://github.com/lensacom/sparkit-learn',
    packages=['splearn',
              'splearn.cluster',
              'splearn.decomposition',
              'splearn.feature_extraction',
              'splearn.feature_selection',
              'splearn.linear_model',
              'splearn.svm'],
    long_description=open('../README.md').read(),
    install_requires=open('requirements.txt').read().split()
)
