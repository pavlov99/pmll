import numpy as np
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt

import sys
sys.path.append('..')  # parent directory with library
from pmll import classification

if __name__ == '__main__':
    data = scipy.io.loadmat('../../data/iris.mat')
    x = data['X'][50:]
    y = data['Y'][50:] - 1

    number_random_features = 3
    x = np.hstack([x, np.random.randn(x.shape[0], number_random_features)])

    model_irls = classification.IrlsModel()
    model_irls.train(x, y, regularization=1e-3, max_iterations=500)

    plt.plot(np.hstack(model_irls._IrlsModel__history['weights']).T)
    plt.plot(model_irls._IrlsModel__history['weight_change'])
    plt.show()

    classifier_irls = classification.IrlsClassifier(model_irls)
    print classifier_irls.classify(x)
