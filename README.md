# Python machine learning library - pmll

[![Coverage Status](https://coveralls.io/repos/pavlov99/pmll/badge.png)](https://coveralls.io/r/pavlov99/pmll)

[![Downloads](https://pypip.in/v/pmll/badge.png)](https://crate.io/packages/pmll)
[![Downloads](https://pypip.in/d/pmll/badge.png)](https://crate.io/packages/pmll)

mater: [![Build Status](https://travis-ci.org/pavlov99/pmll.png?branch=master)](https://travis-ci.org/pavlov99/pmll)

develop: [![Build Status](https://travis-ci.org/pavlov99/pmll.png?branch=develop)](https://travis-ci.org/pavlov99/pmll)

Inspired by: R, Matlab, orange

Author: Kirill Pavlov <mailto:kirill.pavlov@phystech.edu>

Documentation: [https://pmll.readthedocs.org](https://pmll.readthedocs.org)

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

| Title | Description | Established | License | #Developers | #codelines |
|-------|-------------|:-----------:|---------|:-----------:|------------|
| [bigml](https://bigml.com/) [[code]](https://github.com/bigmlcom/python) | Machine learning easy by taking care of the details required to add data-driven decisions and predictive power to your company. | 2012-05-03 | Apache-2.0| [6](https://github.com/bigmlcom/python/blob/next/CONTRIBUTORS) | |
| [mdp](http://mdp-toolkit.sourceforge.net/) [[code]](https://github.com/mdp-toolkit/mdp-toolkit) | Modular toolkit for Data Processing. | 2005-07-20 | BSD [updated](http://mdp-toolkit.sourceforge.net/license.html) | [5](http://mdp-toolkit.sourceforge.net/development.html)
| [milk](http://luispedro.org/software/milk) [[code]](https://github.com/luispedro/milk/) | Machine Learning Toolkit for Python. | 2008-10-14 | MIT| [6](https://github.com/luispedro/milk/contributors) | |
| [mlpy](http://mlpy.sourceforge.net/) [[code]](http://sourceforge.net/p/mlpy/code/ci/default/tree/)| Python module for Machine Learning built on top of NumPy/SciPy and the GNU Scientific Libraries. | 2011-10-11 | GPLv3 | [6](http://mlpy.sourceforge.net/) | |
| [orange](http://orange.biolab.si/) [[code]](https://bitbucket.org/biolab/orange/src) | Data mining through visual programming or Python scripting. | 2003-03-21 | GPLv3 | 122 | |
| [pybrain](http://pybrain.org/) [[code]](https://github.com/pybrain/pybrain)| Modular Machine Learning Library for Python. Mainly networks. | 2008-04-11 | BSD | [15](https://github.com/pybrain/pybrain/contributors)
| [pyml](http://pyml.sourceforge.net/) | Interactive object oriented framework for machine learning written in Python. PyML focuses on SVMs and other kernel methods. | 2010-06-17 | LGPLv2 | [1](http://sourceforge.net/p/pyml/wiki/Home/) | |
| [PyMVPA](http://www.pymvpa.org/) [[code]](https://github.com/PyMVPA/PyMVPA) | Multivariate Pattern Analysis in Python. | 2007-05-23 | MIT | [14](https://github.com/PyMVPA/PyMVPA/contributors)
| [scikit](http://scikit-learn.org/stable/) [[code]](https://github.com/scikit-learn/scikit-learn)| Machine learning in Python. | 2010-01-05 | BSD 3-Clause |[100](https://github.com/scikit-learn/scikit-learn/contributors)
| [Shogun](http://shogun-toolbox.org/) [[code]](https://github.com/shogun-toolbox/shogun) | Mainly SVM kernel methods. C++ and wrappers. machine learning powered by Vodka, Beer and Mate. | 2006-06-05 | GPLv3 | [56](https://github.com/shogun-toolbox/shogun/contributors) | |

* Machine learning libraries at [sourceforge](http://sourceforge.net/directory/science-engineering/ai/machinelearning/os:mac/freshness:recently-updated/).
* [Machine learning open source software](http://mloss.org/software/).
* [Related projects by Shogun](http://shogun-toolbox.org/page/about/related)

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

## Launchpad
https://launchpad.net/pmll


[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/pavlov99/pmll/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

