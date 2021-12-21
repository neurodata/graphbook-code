#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["graspologic", "seaborn", "matplotlib", "statsmodels", 'hyppo', 'numpy>=1.8.0', 'FisherExact']

setup_requirements = [
    "pytest-runner",
    "numpy>=1.8.0",
]

test_requirements = [
    "pytest",
]

setup(
    author="Alex Loftus",
    author_email="aloftus2@jhu.edu",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Python Boilerplate contains all the boilerplate you need to create a Python package.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="graphbook_code",
    name="graphbook_code",
    packages=find_packages(include=["graphbook_code"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/loftusa/graphbook_code",
    version="0.1.0",
    zip_safe=False,
)
