from setuptools import setup, find_packages
setup(
    name="pmll",
    version="0.1",
    packages=find_packages(),
    test_suite="nose2.collector.collector",

    # metadata for upload to PyPI
    author="Kirill Pavlov",
    author_email="kirill.pavlov@phystech.edu",
    description="Python machine learning library",
    license="MIT",
)
