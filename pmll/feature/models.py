# -*- coding: utf-8 -*-
import sympy

__author__ = "Kirill Pavlov"
__email__ = "kirill.pavlov@phystech.edu"


class FeatureMeta(type):
    """MetaClass for Features

    For each feature defines its scala during class creation. Adds classmethod
    to convert value according to feature type
    """
    __store__ = dict()

    def __new__(cls, name, bases, attrs):
        scale = name[len("Feature"):].lower()
        class_ = super(FeatureMeta, cls).__new__(cls, name, bases, attrs)
        setattr(class_, "scale", scale)
        setattr(class_, "convert", classmethod(
            lambda cls, x: Feature.FEATURE_TYPE_MAPPER.get(
                scale, Feature.DEFAULT_TYPE)(x)))
        cls.__store__[scale] = class_
        return class_


class Feature(object):
    """Feture representation
    Converts feature value according to one of the following types:

    nom:  nominal value represented by string
    lin:  float number in linear scale
    rank: float number, arithmetic operations are not supported
    bin:  binary format, true/false or 1/0

    Feature does not know about data, does not have any mean or deviation.
    """
    __metaclass__ = FeatureMeta

    FEATURE_TYPE_MAPPER = {
        "nom": str,
        "lin": float,
        "rank": float,
        "bin": bool,
    }
    DEFAULT_SCALE = "nom"
    DEFAULT_TYPE = str

    def __init__(self, title, scale=DEFAULT_SCALE):
        """Init Feature class

        scale:      feature scale, defines result type and operations allowed
                    for feature.
        formula:    sympy.core.Expr - symbolic expression with feature formula
                    for atom features it is basic itemgetter and transform to
                    type, according to feature class. For not atom features
                    formula consist of expression with atom features as
                    variables.
        convert_atoms {"feature_title": feature} allows complicated feature
                    calculation.
        """
        self.title = str(title)
        self.formula = sympy.Symbol(title)
        self.scale = self.scale or scale
        self._atoms_map = {title: self}

    @property
    def proxy(self):
        return self.__class__.__store__[self.scale](self.title)

    def __str__(self):
        return str(self.formula)

    def __eq__(self, other):
        return not(self < other or other < self)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        """Helper method to order features"""
        return (self.scale, self.title) < (other.scale, other.title)

    def __hash__(self):
        return hash((self.scale, self.title))

    def __call__(self, objects):
        from ..data import Data

        if isinstance(objects, Data):
            # TODO: calculate features not in Data elementwise
            return objects[:, self]
        else:
            if self.formula.is_Atom:
                result = getattr(objects, self.title)
            else:
                subs = {
                    str(arg): self._atoms_map[str(arg)](objects)
                    for arg in self.formula.atoms()
                    if isinstance(arg, sympy.Symbol)}
                result = self.formula.subs(subs)

            return self.convert(result)


class FeatureNom(Feature):
    pass


class FeatureLin(Feature):
    def __neg__(self):
        f = FeatureLin("")
        f.formula = -self.formula
        f._atoms_map.update(self._atoms_map)
        f.title = str(f.formula)
        return f

    def __add__(self, other):
        f = FeatureLin("")
        f._atoms_map.update(
            dict(self._atoms_map.items() + other._atoms_map.items()))
        f.formula = self.formula + other.formula
        return f

    def __sub__(self, other):
        f = FeatureLin("")
        f._atoms_map.update(
            dict(self._atoms_map.items() + other._atoms_map.items()))
        f.formula = self.formula - other.formula
        return f

    def __mul__(self, other):
        f = FeatureLin("")
        f._atoms_map.update(
            dict(self._atoms_map.items() + other._atoms_map.items()))
        f.formula = self.formula * other.formula
        return f

    def __div__(self, other):
        f = FeatureLin("")
        f._atoms_map.update(
            dict(self._atoms_map.items() + other._atoms_map.items()))
        f.formula = self.formula / other.formula
        return f

    def __mod__(self, other):
        return NotImplementedError("Feature is not implemented yet")

    def __divmod__(self, other):
        return NotImplementedError("Feature is not implemented yet")

    def __pow__(self, other, modulo=None):
        f = FeatureLin("")
        f._atoms_map.update(
            dict(self._atoms_map.items() + other._atoms_map.items()))
        f.formula = self.formula ** other.formula
        return f


class FeatureBin(Feature):
    def __and__(self, other):
        f = FeatureBin("")
        f._atoms_map.update(
            dict(self._atoms_map.items() + other._atoms_map.items()))
        f.formula = self.formula & other.formula
        return f

    def __xor__(self, other):
        f = FeatureBin("")
        f._atoms_map.update(
            dict(self._atoms_map.items() + other._atoms_map.items()))
        f.formula = self.formula ^ other.formula
        return f

    def __or__(self, other):
        f = FeatureBin("")
        f._atoms_map.update(
            dict(self._atoms_map.items() + other._atoms_map.items()))
        f.formula = self.formula | other.formula
        return f


class FeatureRank(Feature):
    pass
