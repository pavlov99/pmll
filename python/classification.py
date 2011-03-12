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


class BinaryClassifierBase(BaseClassifier):
    """
    Base class for binary classification
    """
    def __init__(self):
        pass

    number_classes = 2
    cut_off = 0.5

    def get_singular_errors(self, labels, probabilities):
        """
        Return share of errors (unequal objects)
        """
        labels, probabilities = np.asarray(labels), np.asarray(probabilities)
        return float(sum(labels != (probabilities > self.cut_off))) / len(labels)

    def get_auc(labels, probabilities):
        """
        Calculate Area Under Curve for classification.
    
        Arguments:
        - `labels`: class labels in {0, 1}
        - `probabilities`: probabilities of "1" class
    
        Example:
            size = 100000
            labels = numpy.asarray(np.random.randint(0, 2, size))
            import random
            probabilities = [random.random() for i in xrange(size)]

            auc, fpr, tpr = get_auc(labels, probabilities)
            import pylab as p
            p.plot(fpr, tpr)
            p.show()

        >>> get_auc([-1, 1], [0, 0])
        Traceback (most recent call last):
            ...
        ValueError: labels contains not only {0, 1}
        >>> get_auc([0, 1], [0, 0, 1])
        Traceback (most recent call last):
            ...
        AssertionError: lists has different lengths
        >>> get_auc([0, 0, 1, 1], [0.0, 0.6, 0.4, 0.8])
        (0.75, [0, 0, 0.5, 0.5, 1.0], [0, 0.5, 0.5, 1.0, 1.0])
        """
        labels, probabilities = np.asarray(labels), np.asarray(probabilities)
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
    
    
class LeastSquaresBinaryClassifier(BinaryClassifierBase):
    weights = None
    
    def train(self, data_set):
        x, y = data_set.objects.objects, data_set.labels
        self.weights = (x.T * x) ** (-1) * x.T * y 

    def classify(self, object_feature_matrix):
        x = object_feature_matrix.objects
        predicted_labels = x * self.weights
        return predicted_labels
    
    
class RandomBinaryClassifier(BinaryClassifierBase):
    
    
    def train(self, data_set):
        pass

    def classify(self, object_feature_matrix):
        import random
        
        if type(object_feature_matrix) is not ObjectFeatureMatrix:
            raise AssertionError("object`s type is not ObjectFeatureMatrix")
            
        return [random.randint(0, 1) for i in xrange(object_feature_matrix.nobjects)]










