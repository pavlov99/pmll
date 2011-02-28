import numpy as np

class ObjectFeatureMatrix(object):
    def __init__(self, matrix):
        self.__objects = np.asarray(matrix)
        self.__number_objects, self.__number_features = self.__objects.shape
    
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
            x, y = np.asmatrix(x), np.asmatrix(y).T
            return np.asarray(y - x * (x.T * x)**(-1) * x.T * y)
        
        vif = np.empty([self.__number_features, 1])
        for i in xrange(self.__number_features):
            rows = range(i) + range(i + 1, self.__number_features)
            vif[i] = sum((self.__objects[:, i] - np.mean(self.__objects[:, i]))** 2) /\
            sum(_regression_residuals(self.__objects[:, rows], self.__objects[:, i]) ** 2)

        return vif
        
    def get_number_objects(self): 
        return self.__number_objects
    nobjects = property(get_number_objects, doc="return number of objects")
    
    def get_number_features(self): 
        return self.__number_features
    nfeatures = property(get_number_features, doc="return number of features")

        
    

x = [[1, 2], [5, 7], [12, 1]]
d = ObjectFeatureMatrix(x)
print d.vif()
print "\n".join(dir(d))
