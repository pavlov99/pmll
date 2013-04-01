import os
from setuptools import setup, find_packages
from pmll import version


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except IOError:
        return ''

install_requires = read('requirements.txt').split()

setup(
    name="pmll",
    version=version,
    packages=find_packages(),
    test_suite="nose2.collector.collector",
    tests_require=['nose2'],

    # metadata for upload to PyPI
    author="Kirill Pavlov",
    author_email="kirill.pavlov@phystech.edu",
    url="https://github.com/pavlov99/pmll",
    description="Python machine learning library",
    keywords="data mining",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
    ],
    license="MIT",
)
