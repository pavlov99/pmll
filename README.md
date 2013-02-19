# Python machine learning library - pmll

[build_status]: https://api.travis-ci.org/pavlov99/pmll.png  "pmll build status"

Inspired by: R, Matlab, orange

Author: Kirill Pavlov <mailto:kirill.pavlov@phystech.edu>


## Data Format:
Data is stored in tab separated file. First line is header with field names and types. First line starts with sharp and space (# ). Then follows label:label_type [field:type]


Possible scale(field) types:

* lin
* bin
* nom
* rank

Example:

    # label:nom	weight:lin	heigth:lin
    0	70	180


## Naming convention:
Because of many matrix manipulation (multiplication and inversion) variables have short names. Use
x - object-feature matrix
y - labels
w - weights, model parameters

Dont use capitaized variables, it prevent collisions with class names