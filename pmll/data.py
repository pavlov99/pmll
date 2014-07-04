# -*- coding: utf-8 -*-
from collections import namedtuple
import itertools
import numpy as np
import random

from . import six
from .feature import Feature
from .utils import cached_property


class Data(object):

    """ General data representation.

    It is object-feature matrix. There is no label, all of the features are
    equal. It is job for data manager to define what is label.

    Attributes:
        features    list of features. If features are not provided, they are
                    generated as Nominal (string type).
        objects     np.array of objects. Objects could consist of strings, so
                    they are not matrix, they dont support matrix operations.

    There are operation to extend current instance:
        + (__add__)     adds features. It is commutative, feature order does
                        not matter
        .extend(Data)   adds elements with the same features. Object order
                        could matter.
                        TODO: may be easier create new data?
    """

    def __init__(self, objects, features=None):
        """ Init data class.

        :param list objects: convertable to list instances
        :param list features: list of Features

        """
        if features is not None and len(features) > len(set(features)):
            raise ValueError("Features are intersected, but should be unique")

        objects = list(objects)
        self.features = features or \
            [Feature("f{0}".format(i)).proxy for i in range(len(objects[0]))]

        self._objects = [tuple(obj) for obj in objects]
        self.nfeatures = len(self.features)

    def __repr__(self):
        return "Features: {0}\n{1}".format(
            " ".join([str(f) for f in self.features]),
            self.objects.__repr__())

    @cached_property
    def matrix(self):
        """Return matrix of objects if features are linear."""
        if not all(f.scale == "lin" for f in self.features):
            raise ValueError("Could convert only for lenear features")
        return np.matrix(self.objects.tolist())

    @property
    def objects(self):
        fdtype = lambda f: (f.title, ) + Feature.FEATURE_TYPE_MAP[f.scale]
        dtype = np.dtype([fdtype(f) for f in self.features])
        return np.array(self._objects, dtype=dtype)

    @property
    def nobjects(self):
        return len(self._objects)

    def __eq__(self, other):
        """ Check equality of datasets.

        data1 == data2 if they have the same features and
        objects being sorted according to data1.features are equal.
        :return bool:

        """
        l1 = list(zip(*[list(self.objects[f.title])
                        for f in sorted(self.features)]))
        l2 = list(zip(*[list(other.objects[f.title])
                        for f in sorted(other.features)]))
        return set(self.features) == set(other.features) and l1 == l2

    def __ne__(self, other):
        return not (self == other)

    def __getitem__(self, key):
        # TODO: Add feature sclice (given list of features return Data)
        if not isinstance(key, tuple):
            key = (key, slice(None, None, None))

        # Convert second index to slice
        if isinstance(key[1], Feature):
            key = (key[0], self.features.index(key[1])) + key[2:]

        if isinstance(key[1], int):
            key = (key[0], slice(key[1], key[1] + 1)) + key[2:]

        features = self.features.__getitem__(key[1])

        objects = self.objects.__getitem__(key[0]).tolist()
        Object = namedtuple('Object', [f.title for f in features])

        if isinstance(key[0], int):
            return Object(*objects)
        else:
            objects = [list(o).__getitem__(key[1]) for o in objects]
            return Data(objects, features)

    def __add__(self, other):
        if self.nobjects != other.nobjects:
            raise ValueError("Number of objects should be equal")

        objects = [list(o1) + list(o2) for o1, o2
                   in zip(self.objects, other.objects)]
        return Data(objects, features=self.features + other.features)

    @property
    def vif(self):
        """Calculate variance inflation factor"""
        if self.nfeatures < 2:
            raise ValueError("Objects should have at least 2 features")

        if self.nobjects < self.nfeatures:
            raise ValueError("Number of objects should be more than features")

        def __regression_residuals(x, y):
            """Calculate regression residuals:
            y - x * (x' * x)^(-1) * x' * y;
            Input:
                x - np.matrix(l, n)
                y - np.matrix(l, 1)
            Output:
                residuals y - x*w : array(l, 1)
            """
            return np.asarray(y - x * (x.T * x) ** (-1) * x.T * y)

        objects, vif = self.matrix, []
        for i in range(len(self.features)):
            rows = list(range(i)) + list(range(i + 1, len(self.features)))
            residuals = __regression_residuals(objects[:, rows], objects[:, i])
            v = objects[:, i].std() ** 2 / sum(residuals ** 2)
            vif.append(float(v))

        return vif

    @property
    def stat(self):
        return {
            feature: feature.getstat(self.objects[feature.title])
            for feature in self.features}

    @classmethod
    def split(cls, data, ratio=None, size=None):
        """ Split data objects into two groups.

        .. versionadded:: 0.1.4

        Method is used to split data into train and test sets.
        Specify either ratio or size.

        """
        if size is not None:
            predicate = [True] * size + [False] * (data.nobjects - size)
            random.shuffle(predicate)
        else:
            predicate = (random.random() < ratio for i in range(data.nobjects))

        objs1, objs2 = [], []
        for p, obj in zip(predicate, data.objects):
            if p:
                objs1.append(obj)
            else:
                objs2.append(obj)

        return (
            Data(objs1, features=data.features),
            Data(objs2, features=data.features),
        )

    def get_autoregression_data(self, feature, period):
        """ Get autocorrelation matrix for given feature.

        .. versionadded:: 0.1.6
        """
        values = [o[0] for o in self[:, feature].objects]
        data = Data([
            values[i:i + period] for i in range(len(values) - period + 1)
        ])
        return data


class DataReader(object):

    """Read data form tab separated file stream.
    Read data either into objects or matri. Stream can be open(file) or line
    generator.
    """

    HEADER_PREFIX = "# "

    @classmethod
    def __parse_header(cls, header):
        if not header.startswith(cls.HEADER_PREFIX):
            msg = "Bad header format. Should starts from {0}".format(
                cls.HEADER_PREFIX)
            raise ValueError(msg)
        else:
            header = header[len(cls.HEADER_PREFIX):].split('#', 1)[0].rstrip()
            header = header.replace(",", "\t").replace(";", "\t")

        features = [
            Feature(**dict(zip(["title", "scale"], field.split(':')))).proxy
            for field in header.split("\t")]

        duplicated_features = cls.__get_duplicated_features(features)
        if duplicated_features:
            msg = "Duplicated features passed: %s" % duplicated_features
            raise ValueError(msg)

        return features

    @classmethod
    def __get_duplicated_features(cls, features):
        """Return list of duplicated features"""
        if len(set(features)) != len(features):
            return [f for f in features if features.count(f) > 1]

    @classmethod
    def get_objects_features(cls, stream, delimiter="\t"):
        """Read tab separated values.
        Return features and object generator
        """
        # convert stream to generator
        stream = (line for line in stream)
        header = six.advance_iterator(itertools.islice(stream, 1))
        features = cls.__parse_header(header)

        Object = namedtuple('Object', [f.title for f in features])
        objects = (Object(*[feature.convert(value) for feature, value
                            in zip(features, line.strip().split(delimiter))])
                   for line in stream)

        return objects, features

    @classmethod
    def read(cls, stream, delimiter="\t"):
        return Data(*cls.get_objects_features(stream, delimiter=delimiter))
