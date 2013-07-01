# -*- coding: utf-8 -*-
import numpy as np
import scipy
from scipy.io import loadmat
from collections import namedtuple
import itertools
import tempfile
import settings
import urllib2
import gzip
from StringIO import StringIO
from scipy.linalg import svd
import random
import unittest

__author__  = "Kirill Pavlov"
__email__   = "kirill.pavlov@phystech.edu"


class Feature(object):
    """
    Feture representation, converts feature value according to one of the
    following types:

    nom: nominal value represented by string
    lin: float number in linear scale
    rank: float number, arithmetic operations are not supported
    bin: binary format, true/false or 1/0
    """
    FEATURE_TYPE_MAPPER = {
        'nom': str,
        'lin': float,
        'rank': float,
        'bin': bool,
        }

    def __init__(self, title, type_=None):
        self.title = unicode(title)
        self.type = type_
        self.convert = self.FEATURE_TYPE_MAPPER.get(self.type, str)

    def __str__(self):
        return unicode(self).encode('utf8')

    def __unicode__(self):
        return "%s:%s" % (unicode(self.title), unicode(self.type))

    def __eq__(self, other):
        return self.title == other.title and self.type == other.type


class ObjectFeatureMatrix(np.matrix):
    """
    Class for representation of objects-by-features matrix of features.
    """
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

    @property
    def nobjects(self):
        return self.shape[0]

    @property
    def nfeatures(self):
        return self.shape[1]

    @property
    def vif(self):
        """
        Calculate variance inflation factor
        """
        if (self.nfeatures < 2):
            raise ValueError("Objects should have at least 2 features")
        def __regression_residuals(x, y):
            """
            Calculate regression residuals:
            y - x * (x' * x)^(-1) * x' * y;
            Input:
                x - array(l, n)
                y - array(l, 1)
            Output:
                residuals y - x*w : array(l, 1)
            """
            return np.asarray(y - x * (x.T * x)**(-1) * x.T * y)

        vif = []
        for i in range(self.nfeatures):
            rows = range(i) + range(i + 1, self.nfeatures)
            residuals = __regression_residuals(self[:, rows], self[:, i])
            v = self[:, i].std() ** 2 / sum(residuals ** 2)
            vif.append(float(v))

        return vif

    @property
    def belsley(self):
        [U,S,V] = svd(self);

        # divide each row of V.^2 by S(i).^2 - singular value
        Q = np.dot(np.diag((1 / S) ** 2), V ** 2)

        # Normalize Q: column total is 1
        Q = np.dot(Q, np.diag(1 / sum(Q)))

        conditionality_indexes = max(S) / S[:,np.newaxis]
        return(conditionality_indexes, Q)


class ObjectFeatureMatrixTest(unittest.TestCase):
    def setUp(self):
        self.x = ObjectFeatureMatrix([[0, 1], [1, 2], [3, 4]])

    def test_init(self):
        self.assertTrue(isinstance(self.x, ObjectFeatureMatrix))

    def test_nfeatures(self):
        self.assertEqual(self.x.nfeatures, 2)

    def test_nobjects(self):
        self.assertEqual(self.x.nobjects, 3)

    def test_vif(self):
        pass


class Object(object):
    pass


class Sample(object):
    """
    Object in general, consist of all of its characteristics including label
    """
    pass


class DataTest(unittest.TestCase):
    def setUp(self):
        self.nobjects = 4
        self.nfeatures = 3
        self.x = np.random.randint(0, 10, (self.nobjects, self.nfeatures))

    #     self.data = Data(x, y)
    #     Data(x, y, features)

    # d[index] -> object(x, y)
    # d[indexes] -> Data with selected indexes
    # d[:] -> d
    # d[:, index(es)] -> corresponding feature(s) = Data

class FeatureNominal(Feature):
    """
    Store possible values for objects and mapper them to integers
    """
    pass


class Data(object):
    """
    General data representation: objects, labels and features
    objects: ObjectFeatureMatrix
    labels: None of ObjectFeatureMatrix
    features: list<Feature>
    """
    def __init__(self, objects, labels=None, features=None):


        # BUG! ObjectFeatureMatrix(objects) is matrix not ObjectFeatureMatrix!

        # print type(ObjectFeatureMatrix(objects))
        self.x = ObjectFeatureMatrix(objects)
        self.features = features

        # print len(features)
        # print type(self.x) # , self.x

        if self.features is not None and len(features) != self.x.nfeatures:
            msg = "Number of features and columns in objects should be equal"
            # raise ValueError(msg)

        if labels is not None:
            self.y = np.matrix(labels)
        else:
            self.y = np.matrix([[None]] * len(features))

    def __getitem__(self, key):
        print type(key), key
        return self

    @property
    def nfeatures(self):
        return len(self.features)

    def __unicode__(self):
        return unicode(self.x)

    def __str__(self):
        return unicode(self).encode('utf8')


class BigData(object):
    """
    Big data representation. If it is not possible to load data into memory
    one can use iterator technique. It allows to calculate base statistics
    on data and later use chuncks precessing
    """
    pass


class DataReader(object):
    """
    Read data form tab separated file stream either into objects or matrix
    stream can be open(file) or line generator
    """
    def __init__(self):
        self.objects = None
        self.features = None

    def __parse_header(self, header):
        heared_prefix = "# "
        if not header.startswith(heared_prefix):
            msg = 'Bad header format. Should starts from "%s"' % heared_prefix
            raise ValueError(msg)
        else:
            header = header[len(heared_prefix):].split('#', 1)[0].rstrip()
            header = header.replace(",", "\t").replace(";", "\t")

        self.features = [Feature(field.split(':')[0], field.split(':')[1])
                         for field in header.split("\t")]

        return self.features


    def __get_nominal_features_mapper(self, stream, features):
        """
        Convert nominal features (also strings) to integer numbers.
        """
        return stream

    def read(self, stream, delimiter="\t"):
        """
        read tab separated values
        """
        header = itertools.islice(stream, 1).next()
        features = self.__parse_header(header)

        try:
            label_index = [f.title for f in features].index('label')
            feature_indexes = range(label_index) +\
                range(label_index + 1, len(features))
        except ValueError:
            msg = 'Feature header should consist of "label". Current ' +\
                'features are:\n%s' % "; ".join(map(str, features))
            raise ValueError(msg)

        Object = namedtuple('Object', [f.title for f in features])
        objects = (Object(*[feature.convert(value) for feature, value
                            in zip(features, line.strip().split(delimiter))])
                   for line in stream)
        # convert nominal features to integers


        # create data for normalized elements
        self.objects = np.matrix([list(obj) for obj in objects])

        data = Data(
            self.objects[:, feature_indexes],
            labels=self.objects[:, label_index],
            features=features,
            )
        return data


class DataStorage(object):
    def __init__(self):
        self.__fields = None

    def load(self, dataset_name):
        """
        Load dataset for given name
        """
        if not dataset_name in settings.DATA_SETS:
            raise ValueError("Dataset does not exist")
        else:
            self.__datastream = tempfile.TemporaryFile()
            dataset_url = "%s/%s.tsv.gz" % (settings.DATA_HOST, dataset_name)
            response = urllib2.urlopen(dataset_url, timeout=10)
            for line in gzip.GzipFile(fileobj=StringIO(response.read())):
                self.__datastream.write(line)

    @property
    def lines(self):
        self.__datastream.seek(0)
        for line in self.__datastream:
            yield line

    def __get_fields(self):
        header = itertools.islice(self.lines, 1).next()
        heared_prefix = "# "
        if not header.startswith(heared_prefix):
            msg = 'Bad header format. Should starts from "%s"' % heared_prefix
            raise ValueError(msg)
        else:
            header = header[len(heared_prefix):].split('#', 1)[0].rstrip()
            header = header.replace(",", "\t").replace(";", "\t")

        fields = [Field(field.split(':')[0], field.split(':')[1])
                  for field in header.split("\t")]
        return fields

    @property
    def fields(self):
        if self.__fields is None:
            self.__fields = self.__get_fields()
        return self.__fields


if __name__ == "__main__":
    unittest.main()
