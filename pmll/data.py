# -*- coding: utf-8 -*-
from collections import namedtuple
import itertools
import numpy as np


__author__ = "Kirill Pavlov"
__email__ = "kirill.pavlov@phystech.edu"


class Feature(object):
    """
    Feture representation, converts feature value according to one of the
    following types:

    nom: nominal value represented by string
    lin: float number in linear scale
    rank: float number, arithmetic operations are not supported
    bin: binary format, true/false or 1/0

    Feature does not know about data, does not have any mean or deviation.
    """
    DEFAULT_SCALE = "null"
    DEFAULT_TYPE = str
    FEATURE_TYPE_MAPPER = {
        "nom": str,
        "lin": float,
        "rank": float,
        "bin": bool,
    }
    SCALES = FEATURE_TYPE_MAPPER.keys()

    def __init__(self, title, scale=None):
        """Init Feature object

        __is_atom = True for base features
                    False for features created using others
        """
        self.title = title
        self.scale = scale or self.DEFAULT_SCALE
        self.convert = self.FEATURE_TYPE_MAPPER.get(
            self.scale, self.DEFAULT_TYPE)
        self.__is_atom = True

    def __str__(self):
        return unicode(self).encode('utf8')

    def __unicode__(self):
        return "%s:%s" % (unicode(self.title), unicode(self.scale))

    def __eq__(self, other):
        return not(self < other or other < self)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        """Helper method to order features"""
        return (self.scale, self.title) < (other.scale, other.title)

    def __add__(self, other):
        """
        Return feature which is sum of other linear features
        """
        title = "%s + %s" % (self.title, other.title)
        return self.__class__(title, "lin")


class Data(object):
    """
    Data is general data representation. It is object x feature matrix.
    There is no label, all of the features are equal. It is job for data
    manager to define what is label.
    """
    def __init__(self, objects, features=None):
        """Init data class
        objects: convertable to list instances
        features: list of Features
        """
        self.objects = np.matrix([list(obj) for obj in objects])
        self.features = features or [Feature("f%s" % i)
                                     for i in range(self.objects.shape[1])]

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
        return features_self == features_other and \
                (self.objects[:, np.array(indexes_self)] == \
                other.objects[:, np.array(indexes_other)]).all()

    def __ne__(self, other):
        return not (self == other)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self.__class__(
                    self.objects.__getitem__(key).tolist(), self.features)
        elif len(key) == 2:
            if isinstance(key[1], Feature):
                key = key[0], self.features.index(key[1])

            objects = self.objects.__getitem__(key).tolist()
            if isinstance(key[1], int):
                features = [self.features[key[1]]]
            elif isinstance(key[1], slice):
                features = self.features.__getitem__(key[1])
            return self.__class__(objects, features)


class DataReader(object):
    """
    Read data form tab separated file stream either into objects or matrix
    stream can be open(file) or line generator
    """
    @classmethod
    def __parse_header(cls, header):
        heared_prefix = "# "
        if not header.startswith(heared_prefix):
            msg = 'Bad header format. Should starts from "%s"' % heared_prefix
            raise ValueError(msg)
        else:
            header = header[len(heared_prefix):].split('#', 1)[0].rstrip()
            header = header.replace(",", "\t").replace(";", "\t")

        features = [
            Feature(*field.split(':'))
            for field in header.split("\t")
        ]

        duplicated_features = cls.__get_duplicated_features(features)
        if duplicated_features:
            msg = "Duplicated features passed: %s" % duplicated_features
            raise ValueError(msg)

        return features

    @classmethod
    def __get_duplicated_features(cls, features):
        """
        Return list of duplicated feature titles
        """
        feature_titles = [f.title for f in features]
        if len(set(feature_titles)) != len(features):
            return [f for f in feature_titles if feature_titles.count(f) > 1]

    @classmethod
    def read(cls, stream, delimiter="\t"):
        """
        Read tab separated values.
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
