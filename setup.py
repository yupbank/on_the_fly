#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

from distutils.core import setup

setup(
        name = 'on_the_fly',
        packages = ['on_the_fly',],
        version = "0.02rc3",
        description = """on_the_fly: out-of-core learning for PySpark and Python iterators'

        on_the_fly is a machine learning toolkit designed to efficiently perform
        online feature extraction and learning on RDD and Python iterators.
        """,
        author = 'Peng Yu',
        author_email = 'yupbank@gmail.com',
        url = 'https://github.com/yupbank/on_the_fly',
        download_url = 'https://github.com/yupbank/on_the_fly/archive/0.02rc3.tar.gz',
        keywords = ['on the fly', 'machine learning', 'sklearn', 'online'],
        classifiers = [],
        install_requires = open('requirements.txt').read().split()
        )
