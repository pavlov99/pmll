import numpy as np

class ObjectFeatureMatrix(object):
    """
    Class for representation of objects-by-features matrix of features.

    >>> x = np.array([[1, 2, 3], [4, 6, 87], [3, 5, 68], [3, 4, 6]])
    >>> y = np.array([[1, 2], [4, 6], [3, 5], [3, 4]])
    >>> m = ObjectFeatureMatrix()
    >>> m.nobjects
    0
    >>> m.nfeatures # return nothing
    >>> m.add(x)
    >>> m.nobjects
    4
    >>> m.nfeatures
    3
    >>> m.objects
    matrix([[ 1,  2,  3],
            [ 4,  6, 87],
            [ 3,  5, 68],
            [ 3,  4,  6]])
    >>> m.add(x)
    >>> m.nobjects
    8
    >>> m.nfeatures
    3
    >>> m.add(y)
    Traceback (most recent call last):
        ...
    AssertionError: additional and current objects has different numbers of features
    """
    def __init__(self, matrix=None):
        self.objects = None
        self.nobjects = 0 
        self.nfeatures = None

        if matrix is not None:
            self.add(matrix)

    def __str__(self):
        print self.objects

    def add(self, matrix):
        matrix = np.asmatrix(matrix)
        if self.nfeatures is not None:
            if matrix.shape[1] != self.nfeatures:
                raise AssertionError("additional and current objects has different numbers of features")
            self.objects = np.vstack([self.objects, matrix])
            self.nobjects += matrix.shape[0]
        else: # first add
            self.objects = matrix 
            self.nobjects, self.nfeatures = self.objects.shape

    def vif(self):
        if self.objects is None:
            raise ValueError("Can`t use this method for empty data")

        def _regression_residuals(x, y):
            """
            Calculate regression residuals:
            y - x * (x' * x)^(-1) * x' * y;
            Input:
                x - array(l, n)
                y - array(l, 1)
            Output:
                residuals y - x*w - array(l, 1)
            """
            return np.asarray(y - x * (x.T * x)**(-1) * x.T * y)

        vif = np.empty([self.nfeatures, 1])
        for i in xrange(self.nfeatures):
            rows = range(i) + range(i + 1, self.nfeatures)
            vif[i] = sum(np.asarray(self.objects[:, i] - np.mean(self.objects[:, i])) ** 2) /\
            sum(_regression_residuals(self.objects[:, rows], self.objects[:, i]) ** 2)

        return vif

    def belsley(self):
        if self.objects is None:
            raise ValueError("Can`t use this method for empty data")

        from scipy.linalg import svd

        [U,S,V] = svd(self.objects);

        # divide each row of V.^2 by S(i).^2 - singular value
        Q = np.dot(np.diag((1 / S) ** 2), V ** 2)

        # Normalize Q: column total is 1
        Q = np.dot(Q, np.diag(1 / sum(Q)))

        conditionality_indexes = max(S) / S[:,np.newaxis]
        return(conditionality_indexes, Q)


class DataSet(object):
    def __init__(self, objects=None, labels=None):        
        if (objects is None) ^ (labels is None):
            raise TypeError("Require zero or two arguments for input")

        self.objects = ObjectFeatureMatrix()
        self.labels = None

        if objects is not None: # so labels is not None too
            self.add(objects, labels)

    def add(self, objects, labels):
        objects, labels = np.asmatrix(objects), np.asmatrix(labels)

        if labels.shape[0] == 1:
            labels = labels.T

        if labels.shape[1] != 1:
            raise AssertionError("One of the label dimensions must be 1")

        if labels.shape[0] != objects.shape[0]:
            raise AssertionError("Number of objects must be equal number of labels")

        self.labels = np.vstack([self.labels, labels])
        self.objects.add(objects)

    def get_number_objects(self): 
        return self.objects.nobjects
    nobjects = property(get_number_objects, doc="return number of objects")

    def get_number_features(self): 
        return self.objects.nfeatures
    nfeatures = property(get_number_features, doc="return number of features")


if __name__ == "__main__":
    import doctest
    doctest.testmod()

