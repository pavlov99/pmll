from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from data import ObjectFeatureMatrix, DataSet


class BaseClassifier(object):
    """
    Classification base class
    """

    __metaclass__=ABCMeta
    
    def __init__(self, number_classes):
        self.number_classes = number_classes

    @abstractmethod
    def train(self, data_set): 
        pass
    
    @abstractmethod
    def classify(self, object_feature_matrix): 
        """
        Classification function. Return list or np.array([objects, 1]) of answers
        """
        pass


class BaseBinaryClassifier(BaseClassifier):
    """
    Base class for binary classification
    """
    def __init__(self):
        pass

    number_classes = 2
    cut_off = 0.5

    def get_singular_errors(self, data):
        """
        Return share of errors (unequal objects)
        """
        labels = np.asmatrix(data.labels)
        probabilities = np.asmatrix(self.classify(data.objects))
        return float(sum(labels != (probabilities > self.cut_off))) / len(labels)

    def get_auc(self, data):
        """
        Calculate Area Under Curve for classification.
    
        Arguments:
        - `data`: DataSet object
    
        Example:
            size = 100000
            zeros = np.zeros([size, 1])
            data = DataSet(objects=zeros, labels=zeros)
            
            auc, fpr, tpr = self.get_auc(data)
            import pylab as p
            p.plot(fpr, tpr)
            p.show()
        """

        labels = np.asarray(data.labels.flatten())[0]
        probabilities = np.asarray(self.classify(data.objects))
        
        number_positive = sum(labels == 1)
        number_negative = sum(labels == 0)
    
        if labels.size != probabilities.size:
            raise AssertionError('lists has different lengths')
        if number_positive + number_negative != labels.size:
            raise ValueError('labels contains not only {0, 1}')

        ordered_labels =  zip(probabilities, labels)
        ordered_labels.sort(reverse=True)

        # init values
        fpr = [0] # false positive rate
        tpr = [0] # true positive rate
        auc = 0   # area under curve 
    
        for label in ordered_labels:
            if label[1] == 1:
                fpr.append(fpr[-1])
                tpr.append(tpr[-1] + 1.0 / number_positive)
            else:
                fpr.append(fpr[-1] + 1.0 / number_negative)
                tpr.append(tpr[-1])
                auc = auc + tpr[-1] / number_negative

        return (auc, fpr, tpr)
    
    
class LeastSquaresBinaryClassifier(BaseBinaryClassifier):
    weights = None
    
    def train(self, data_set):
        x, y = data_set.objects.objects, data_set.labels
        self.weights = (x.T * x) ** (-1) * x.T * y 

    def classify(self, object_feature_matrix):
        if self.weights is None:
            raise AssertionError("weights are undefined. Did you train?") 
        x = object_feature_matrix.objects
        predicted_labels = x * self.weights
        return predicted_labels
    
    
class RandomBinaryClassifier(BaseBinaryClassifier):
    def train(self, data_set):
        pass

    def classify(self, object_feature_matrix):
        import random
        
        if type(object_feature_matrix) is not ObjectFeatureMatrix:
            raise AssertionError("object`s type is not ObjectFeatureMatrix")
            
        predictions = np.asarray([random.randint(0, 1) for i in xrange(object_feature_matrix.nobjects)])
        predictions = predictions.reshape(-1, 1)
        return predictions


class Irls(object):
    def __init__(self, objects, labels):
        objects, labels = np.asmatrix(objects), np.asmatrix(labels)
        self.__objects = np.hstack([objects, np.ones([objects.shape[0], 1])])
        self.__labels = labels
        self.__weights = np.zeros([self.__objects.shape[1], 1])
        self.__history = {'weights': self.__weights}

    def __logit(self, z): return 1 / (1 + np.exp(-z))

    def classify(self, objects):
        # conver input to matrix, append constant column and get logit
        return self.__logit(np.asmatrix(np.hstack([objects,
                                                   np.ones([objects.shape[0], 1]
                                                           )])
                                        ) * self.__weights)

    def train(self, max_iterations=100, accuracy=1e-7,
              regularization_parameter=1e-7):
        for iteration in range(max_iterations):
            probability = self.__logit(self.__objects * self.__weights)
            B = np.diagflat(probability - np.power(probability, 2))
            # Dont use -=, it changes history[weights]
            self.__weights = self.__weights - \
                (self.__objects.T * B * self.__objects + \
                 regularization_parameter * np.eye(self.__objects.shape[1]))\
                 ** (-1) * self.__objects.T * (probability - self.__labels)

            self.__history['weights'] = np.hstack([self.__history['weights'],
                                                   self.__weights])

            if sum(abs(self.__history['weights'][:, -1] - \
                           self.__history['weights'][:, -2])) < accuracy:
                break

    def plot_convergence(self):
        from matplotlib import pyplot as plt
        plt.plot(self.__history['weights'].T)
        plt.show()

    def __str__(self): return '%s\n%s' % (str(self.__objects), str(self.__weights))


if __name__ == "__main__":
    import doctest
    doctest.testfile("%s.test" % __file__.split(".", 1)[0])
    # doctest.testfile("%s.test" % __file__.split(".", 1)[0], verbose=True)

