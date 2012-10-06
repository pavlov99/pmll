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


class ObjectFeatureMatrix(np.matrix):
    """
    Class for representation of objects-by-features matrix of features.
    """
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


Feature = namedtuple('Feature', 'type')

class Data(object):
    """
    General data representation: objects, labels and features
    objects: ObjectFeatureMatrix
    labels: None of ObjectFeatureMatrix
    features: list<Feature>
    """
    def __init__(self, objects, labels=None, features=None):
        self.x = ObjectFeatureMatrix(objects)
        self.features = features

        if self.features is not None and len(features) != self.x.nfeatures:
            msg = "Number of features and columns in objects should be equal"
            raise ValueError(msg)

        if labels is not None:
            self.y = np.matrix(labels)
        else:
            self.y = np.matrix([[None]] * len(features))

    @property
    def nfeatures(self):
        return len(self.features)




class BigData(object):
    """
    Big data representation. If it is not possible to load data into memory
    one can use iterator technique. It allows to calculate base statistics
    on data and later use chuncks precessing
    """
    pass



class DataReader(object):
    def read(self, stream, separator="\t"):
        """ read csv """
        header = itertools.islice(stream, 1)
        fields = [Field(field.split(':')[0], field.split(':')[1]) for field in header.split('\t')]

        data_matrix = scipy.io.read_array(stream,
                                          separator="\t",
                                          comment = '#',
                                          )

        data = Data(data_matrix[:,1:],
                    labels=data_matrix[:,0][:,np.newaxis],
                    fields=fields)
        return data


class DataStorage(object):



    # def __init__(self, data, dtype='None', copy='True', features=None):
    #     super(self.__class__, self).__init__(data, dtype=dtype, copy=copy)
    #     self.features = features
    #     if self.features is not None and len(self.features) != self.nfeatures:
    #         msg = "Number of features and matrix columns should be the same"
    #         raise ValueError(msg)



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
    pass
