# -*- coding: utf-8 -*-
from collections import namedtuple
import itertools
import numpy as np

from .feature import Feature

__author__ = "Kirill Pavlov"
__email__ = "kirill.pavlov@phystech.edu"


class Data(object):
    """General data representation.
    It is object-feature matrix. There is no label, all of the features are
    equal. It is job for data manager to define what is label.

    There are operation to extend current instance:
        + (__add__)     adds features. It is commutative, feature order does
                        not matter
        .extend(Data)   adds elements with the same features. Object order
                        could matter.
                        TODO: may be easier create new data?
    """
    def __init__(self, objects, features=None):
        """Init data class
        objects: convertable to list instances
        features: list of Features
        """
        if features and len(features) > len(set(features)):
            raise ValueError("Features should be unique")

        self.objects = np.matrix([list(obj) for obj in objects])
        self.features = features or [
            Feature("f%s" % i) for i in range(self.objects.shape[1])]

    def __repr__(self):
        return "Features: {0}\n{1}".format(
            " ".join([str(f) for f in self.features]),
            self.objects.__repr__())

    def __eq__(self, other):
        """Check equality of datasets

        data1 == data2 if they have the same features and
        objects being sorted according to data1.features are equal
        """
        indexes_self, features_self =\
            zip(*sorted(enumerate(self.features), key=lambda x: x[1]))
        indexes_other, features_other =\
            zip(*sorted(enumerate(other.features), key=lambda x: x[1]))
        return features_self == features_other and (
            self.objects[:, np.array(indexes_self)] ==
            other.objects[:, np.array(indexes_other)]).all()

    def __ne__(self, other):
        return not (self == other)

    def __getitem__(self, key):
        # TODO: Add feature sclice (given list of features return Data)
        features = self.features

        if isinstance(key, tuple):
            # Convert second index to slice
            if isinstance(key[1], Feature):
                key = (key[0], self.features.index(key[1])) + key[2:]

            if isinstance(key[1], int):
                key = (key[0], slice(key[1], key[1] + 1)) + key[2:]

            features = self.features.__getitem__(key[1])

        objects = self.objects.__getitem__(key).tolist()
        Object = namedtuple('Object', [f.title for f in features])

        if isinstance(key, int) or \
           (isinstance(key, tuple) and isinstance(key[0], int)):
            return Object(*objects[0])
        else:
            return Data(objects, features)

    def __add__(self, other):
        if self.objects.shape[0] != other.objects.shape[0]:
            raise ValueError("Number of objects should be equal")

        features = self.features + other.features
        if len(features) > len(set(features)):
            raise ValueError("Features are intersected")

        return Data(np.hstack([self.objects, other.objects]).tolist(),
                    features=features)

    @property
    def vif(self):
        """Calculate variance inflation factor"""
        if len(self.features) < 2:
            raise ValueError("Objects should have at least 2 features")

        if self.objects.shape[0] < self.objects.shape[1]:
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

        vif = []
        for i in range(len(self.features)):
            rows = range(i) + range(i + 1, len(self.features))
            residuals = __regression_residuals(
                self.objects[:, rows], self.objects[:, i])
            v = self.objects[:, i].std() ** 2 / sum(residuals ** 2)
            vif.append(float(v))

        return vif


class DataReader(object):
    """Read data form tab separated file stream.
    Read data either into objects or matri. Stream can be open(file) or line
    generator.
    """
    @classmethod
    def __parse_header(cls, header):
        heared_prefix = "# "
        if not header.startswith(heared_prefix):
            msg = "Bad header format. Should starts from {0}".format(
                heared_prefix)
            raise ValueError(msg)
        else:
            header = header[len(heared_prefix):].split('#', 1)[0].rstrip()
            header = header.replace(",", "\t").replace(";", "\t")

        features = [
            Feature(*field.split(':')).proxy
            for field in header.split("\t")
        ]

        duplicated_features = cls.__get_duplicated_features(features)
        if duplicated_features:
            msg = "Duplicated features passed: %s" % duplicated_features
            raise ValueError(msg)

        return features

    @classmethod
    def __get_duplicated_features(cls, features):
        """Return list of duplicated feature titles"""
        feature_titles = [f.title for f in features]
        if len(set(feature_titles)) != len(features):
            return [f for f in feature_titles if feature_titles.count(f) > 1]

    @classmethod
    def read(cls, stream, delimiter="\t"):
        """Read tab separated values.
        Return features and object generator
        """
        # convert stream to generator
        stream = (line for line in stream)
        header = itertools.islice(stream, 1).next()
        features = cls.__parse_header(header)

        Object = namedtuple('Object', [f.title for f in features])
        objects = (Object(*[feature.convert(value) for feature, value
                            in zip(features, line.strip().split(delimiter))])
                   for line in stream)

        return objects, features
