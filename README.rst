:Authors:
    Kirill Pavlov <kirill.pavlov@phystech.edu>

:Version: 0.1 of 2012/05/28

Data Format:
------------

Data is stored in tab separated file. First line is header with field names and
types. First line starts with sharp and space (# ). Then follows label:label_type [field:type]


Possible scale(field) types:

* lin
* bin
* nom
* rank

Example:
  +------------+------------+------------+
  | label:nom  | weight:lin | heigth:lin |
  +============+============+============+
  |     0      |     70     |     180    |
  +------------+------------+------------+
