#!/usr/bin/env python3


from setuptools import setup, find_packages


setup(
        name="classifier",
        version="0.1.0",
        description="Classification based on images",

        classifiers = [
            "Programming Language :: Python :: 3"
        ],

        packages = find_packages(),

        entry_points = {
            "console_scripts": ["classifier=classifier.__main__:main"],
        },
)
