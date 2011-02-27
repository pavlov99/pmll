import numpy as np

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
    
    
def least_squares(x, y):
    x, y = np.asmatrix(x), np.asmatrix(y)
    print y
    print x.T
    print (x.T * x)^(-1) * x.T 
    # return (x' * x)^(-1) * x' * y
    
    
def get_vif(x):
    """
    Calculate variance inflation factor
    """
    x = np.asarray(x)
    (numer_objects, number_features) = x.shape
    print numer_objects, number_features
    
    pass


def get_belsley(x):
    """
    Calculate ? FIXIT What?
    
    """
    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print 'Done'
    x = np.array([[1, 2], [5, 7], [12, 1]])

    least_squares(x, np.array([[1], [1]]))
    get_vif(x)
    
    
    