# Python machine learning library - pmll

[![Build Status](https://travis-ci.org/pavlov99/pmll.png?branch=master)](https://travis-ci.org/pavlov99/pmll)
[![Downloads](https://pypip.in/d/pmll/badge.png)](https://crate.io/package/pmll)
[![Downloads](https://pypip.in/v/pmll/badge.png)](https://crate.io/package/pmll)

Inspired by: R, Matlab, orange

Author: Kirill Pavlov <mailto:kirill.pavlov@phystech.edu>

Library aimed to bring simplicity to machime learning algorithms usage.

## Installation
To install `pmll` as package, simply run:

    pip install pmll

If you want to develop it, use `make`.

## Tests

    ./setup.py test

or

    nosetests

## Competitors

* [bigml](https://github.com/bigmlcom/python) - machine learning easy by taking care of the details required to add data-driven decisions and predictive power to your company.
* [mdp](http://mdp-toolkit.sourceforge.net/) - Modular toolkit for Data Processing.
* [milk](http://luispedro.org/software/milk) - Machine Learning Toolkit for Python.
* [mlpy](http://mlpy.sourceforge.net/) - is a Python module for Machine Learning built on top of NumPy/SciPy and the GNU Scientific Libraries.
* [orange](http://orange.biolab.si/) - Data mining through visual programming or Python scripting.
* [pybrain](http://pybrain.org/) - is a modular Machine Learning Library for Python. Mainly networks.
* [pyml](http://pyml.sourceforge.net/) - an interactive object oriented framework for machine learning written in Python. PyML focuses on SVMs and other kernel methods.
* [PyMVPA](http://www.pymvpa.org/) - Multivariate Pattern Analysis in Python.
* [scikit](http://scikit-learn.org/stable/) - machine learning in Python.
* [Shogun](http://shogun-toolbox.org/) - machine learning powered by Vodka, Beer and Mate.


## Data Format:
Data is stored in tab separated file. First line is header with field names and types. First line starts with sharp and space (# ). Then follows `label:label_type [field:type]`

Possible scale(field) types:

* nom:  nominal value represented by string
* lin:  float number in linear scale
* rank: float number, arithmetic operations are not supported
* bin:  binary format, true/false or 1/0

Example:

    # label:nom	weight:lin	heigth:lin
    0	70	180
