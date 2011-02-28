import numpy as np

class ObjectFeatureMatrix(object):
    def __init__(self, matrix):
        self.__objects = np.asarray(matrix)
        self.__number_objects, self.__number_features = self.__objects.shape
    
    def vif(self):
        pass
        
    def get_number_objects(self): 
        return self.__number_objects
    nobjects = property(get_number_objects, doc="return number of objects")
    
    def get_number_features(self): 
        return self.__number_features
    nfeatures = property(get_number_features, doc="return number of features")

        
    

x = [[1, 2], [5, 7], [12, 1]]
d = ObjectFeatureMatrix(x)
# print d._ObjectFeatureMatrix__objects
# print d._number_objects
