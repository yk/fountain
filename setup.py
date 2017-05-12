#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(name='fountain',
        version='0.0.1',
        description='Datasets',
        author='yk',
        packages=find_packages(),
        install_requires=['numpy', 'scipy', 'filelock', 'rcfile'],
        )
