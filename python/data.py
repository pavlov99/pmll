import numpy as np

class ObjectFeatureMatrix(object):
    def __init__(self, matrix):
        matrix = np.asmatrix(matrix)
        self.__objects = matrix
        self.__number_objects, self.__number_features = self.__objects.shape
        
    def add_data(self, matrix):
        matrix = np.asmatrix(matrix)
        if matrix.shape[1] != self.__number_features:
            raise AssertionError("Additional and current objects has different \
                numbers of features")
        self.__objects = np.vstack([self.__objects, matrix])
        self.__number_objects += matrix.shape[0]
    
    def vif(self):
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
        
        vif = np.empty([self.__number_features, 1])
        for i in xrange(self.__number_features):
            rows = range(i) + range(i + 1, self.__number_features)
            vif[i] = sum(np.asarray(self.__objects[:, i] - np.mean(self.__objects[:, i])) ** 2) /\
            sum(_regression_residuals(self.__objects[:, rows], self.__objects[:, i]) ** 2)

        return vif

    def belsley(self):
        from scipy.linalg import svd

        [U,S,V] = svd(self.__objects);

        # divide each row of V.^2 by S(i).^2 - singular value
        Q = np.dot(np.diag((1 / S) ** 2), V ** 2)

        # Normalize Q: column total is 1
        Q = np.dot(Q, np.diag(1 / sum(Q)))

        conditionality_indexes = max(S) / S[:,np.newaxis]
        return(conditionality_indexes, Q)

    def get_number_objects(self): 
        return self.__number_objects
    nobjects = property(get_number_objects, doc="return number of objects")
    
    def get_number_features(self): 
        return self.__number_features
    nfeatures = property(get_number_features, doc="return number of features")


class DataSet(object):
    def __init__(self, objects, labels):
        objects, labels = np.asmatrix(objects), np.asmatrix(labels)

        if labels.shape[0] == 1:
            labels = labels.T
        
        if labels.shape[1] != 1:
            raise AssertionError("One of the label dimensions must be 1")
        
        if labels.shape[0] != objects.shape[0]:
            raise AssertionError("Number of objects must be equal number of labels")
        
        self.labels = labels
        self.objects = ObjectFeatureMatrix(objects)
        
