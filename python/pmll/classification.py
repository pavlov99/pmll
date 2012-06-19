from abc import ABCMeta, abstractmethod
import time


def timer():
    def wrapper(f):
        def wrapped_function(*args, **kwargs):
            t = time.time()
            output = f(*args, **kwargs)
            print time.time() - t
            return output


class BaseModel(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.train_time = None
        self.decorate_method('train')

    def decorate_method(self, method_name):
        """
        Decorate given method (change to decoreted):
          - calculate time.
        """
        method = self.decorated_method(getattr(self, method_name))

        def decorated_method(*args, **kwargs):
            # Decorate method: calculate time
            t = time.time()
            output = method(*args, **kwargs)
            setattr(self, '%s_time' % method_name, time.time() - t)
            return output

        setattr(self, method_name, decorated_method)

    @abstractmethod
    def train(self):
        """
        Abstract train method
        """
        pass


class BaseClassifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def classify():
        pass


class ModelLinearRegression(BaseModel):
    def __init__(self, weighs):
        self.weights = weighs

    def train(self, x, y):
        """
        Train Linear regression model y = w * x
        """
        self.weights = (x.T * x).I * x.T * y
        # return classifier?


class BaseModel2(BaseModel):
    pass


class BaseModel3(BaseModel2):
    def train(self, a, b=1, d=[1, 2, 3]):
        return "train in Base Model 3"


def main():
    m = BaseModel3()
    print m.train('a', b=2)
    print m.train_time

if __name__ == '__main__':
    main()
