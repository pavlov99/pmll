import os
from setuptools import setup, find_packages


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except IOError:
        return ''

install_requires = read('requirements.txt').split()

setup(
    name="pmll",
    version=":versiontools:pmll:",
    packages=find_packages(),
    test_suite="nose2.collector.collector",
    setup_requires=[
        'versiontools >= 1.8',
    ],

    # metadata for upload to PyPI
    author="Kirill Pavlov",
    author_email="kirill.pavlov@phystech.edu",
    url="pmll.kirillpavlov.com",
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
