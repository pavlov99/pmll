.. _getting_started:


***************
Getting started
***************

Install
=============================

To install `pmll` use pip::

  pip install pmll


Data structures
===============

There are two main data structures in the package: `feature` and `dataset`.

Feature
-------

Feature class is used to describe objects. Each feature has title and type.

Features are in feature space, they have title and optional type. Type is used to define possible operations with features, it could be `lin`, `nom`, `bin`, `rank`.

.. sourcecode:: ipython

   In [1]: from pmll.feature import Feature, FeatureNom, FeatureLin
   In [2]: from collections import namedtuple
   In [3]: Parallelepiped = namedtuple("Parallelepiped", ["colour", "length", "height", "width"])
   In [4]: cube = Parallelepiped("red", 2, 2, 2)

Lets define features for new object: colour is nominal, others are linear.

.. sourcecode:: ipython

   In [5]: f = FeatureNom("colour")
   In [6]: f(cube)
   Out[6]: 'red'

   In [7]: f1, f2, f3 = FeatureLin("length"), FeatureLin("height"), FeatureLin("width")

It is possible to multiply linear features, for example square equals length times height

.. sourcecode:: ipython

   In [8]: square = f1 * f2
   In [9]: square.title
   Out[9]: 'height*length'

   In [10]: square(cube)
   Out[10]: 4.0

   In [11]: volume = square * f3
   In [12]: volume.title
   Out[12]: 'height*length*width'

   In [13]: volume(cube)
   Out[13]: 8.0
