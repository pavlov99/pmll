import numpy as np
import scipy
import matplotlib.pyplot as plt

import sys
sys.path.append('..')  # parent directory with library
from pmll import classification

if __name__ == '__main__':
    data = scipy.io.loadmat('../../data/iris.mat')
    x = data['X'][50:]
    y = data['Y'][50:] - 1

    number_random_features = 3
    random_feature_names = ['random%s' % i for i
                            in range(1, number_random_features + 1)]
    legend = ['w1', 'w2', 'w3', 'w4'] + random_feature_names + ['1']
    x = np.hstack([x, np.random.randn(x.shape[0], number_random_features)])

    model_irls = classification.IrlsModel()
    model_irls.train(x, y, regularization=1e-3, max_iterations=1000)

    model_gradient = classification.LogisticRegressionGradientModel()
    model_gradient.train2(x, y, regularization=1e-3, max_iterations=1000)

    logit = lambda z: 1 / (1 + np.exp(-z))
    objects = np.matrix(x)
    objects = np.asmatrix(np.column_stack((
                objects,
                np.ones([objects.shape[0], 1]),
                )))
    labels = np.matrix(y)
    q = lambda x, y, w: float(-sum(np.log(logit(np.diagflat(y) * x * w))))

    print q(objects, labels, model_irls.weights)
    print q(objects, labels, model_gradient.weights)
    print model_irls.time
    print model_gradient.time

    ws = model_gradient._LogisticRegressionGradientModel__history['weights']
    plt.plot(np.hstack(ws).T)
    plt.ylabel('Weight')
    plt.xlabel('Iteration')
    plt.legend(legend, loc=3)
    plt.show()

    plt.plot(np.hstack(model_irls._IrlsModel__history['weights']).T)
    # plt.plot(model_irls._IrlsModel__history['weight_change'])
    plt.ylabel('Weight')
    plt.xlabel('Iteration')
    plt.legend(legend, loc=3)
    plt.show()

    # classifier_irls = classification.IrlsClassifier(model_irls)
    # p = classifier_irls.classify(x)
    # auc, fpr, tpr = get_auc(y, p)
    # plt.plot(fpr, tpr)
    # plt.show()
    # print auc
