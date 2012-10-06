import sys
sys.path.append('..')
import numpy as np
from basemodel import BaseModel


class LeastSquaresModel(BaseModel):
    def __init__(self, weights=None):
        super(self.__class__, self).__init__()
        self.weights = weights

    def train(self, x, y):
        self.weights = (x.T * x).I * x.T * y


class LeastSquaresPolynomModel(BaseModel):
    def __init__(self, weights=None):
        super(self.__class__, self).__init__()
        self.weights = weights

    def train(self, x, y, max_power=1, is_add_constant=0):
        # Add polynomial features
        x = np.hstack(np.power(x, power) for power in range(1, max_power + 1))

        if is_add_constant:
            x = np.hstack([np.ones([x.shape[0], 1]), x])

        self.weights = (x.T * x).I * x.T * y


if __name__ == '__main__':
    ls = LeastSquaresModel
    data = np.genfromtxt(open("bread.csv", "rb"), dtype=float, delimiter=',')
    y = np.asmatrix(data[:, 0][:, np.newaxis])
    x = np.asmatrix(data[:, 1][:, np.newaxis])

    # import matplotlib.pyplot as plt
    # plt.plot(x, y, "k.")
    # plt.show()

    model = LeastSquaresPolynomModel()
    for is_add_constant in [0, 1]:
        for max_power in range(1, 10):
            print "Add contsant = %s. Max power = %s" % (is_add_constant,
                                                         max_power)
            model.train(x, y, is_add_constant=1, max_power=max_power)
            print model.weights
            print "Train time:\t", model.train_time

            # import matplotlib.pyplot as plt
            # plt.plot(y, "k.")
            # plt.plot(x * self.weights, "b")
            # plt.show()
            # print "Error:\t\t", float(sum(map(lambda x: x ** 2,
            # y - x * self.weights)))