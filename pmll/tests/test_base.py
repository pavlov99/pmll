import unittest
from ..base import BaseModel, BaseRegressor, BaseClassifier


class BaseModelTest(unittest.TestCase):
    def setUp(self):
        class Model(BaseModel):
            @property
            def predictor(self):
                pass

            def train(self):
                return self

        self.class_ = Model
        self.model = self.class_()

    def test_basemodel_require_train(self):
        BadModel = self.class_
        delattr(BadModel, 'train')
        self.assertRaises(BadModel())

    def test_basemodel_require_predictor(self):
        BadModel = self.class_
        delattr(BadModel, 'predictor')
        self.assertRaises(BadModel())

    def test_basemodel_adds_time_to_train(self):
        self.assertFalse(hasattr(self.model, 'train_time'))
        self.model.train()
        self.assertTrue(hasattr(self.model, 'train_time'))


class BaseClassifierTest(unittest.TestCase):
    def setUp(self):
        class Classifier(BaseClassifier):
            @property
            def classify(self):
                return self

        self.class_ = Classifier
        self.model = self.class_()

    def test_require_classify(self):
        BadModel = self.class_
        delattr(BadModel, 'classify')
        self.assertRaises(BadModel())

    def test_is_callable(self):
        # it is possible to call model
        self.assertTrue(hasattr(self.model, '__call__'))

        # () is link to classify
        self.assertEqual(self.model, self.model.classify)


class BaseRegressorTest(unittest.TestCase):
    def setUp(self):
        class Regressor(BaseRegressor):
            @property
            def regress(self):
                return self

        self.class_ = Regressor
        self.model = self.class_()

    def test_require_regress(self):
        BadModel = self.class_
        delattr(BadModel, 'regress')
        self.assertRaises(BadModel())

    def test_is_callable(self):
        # it is possible to call model
        self.assertTrue(hasattr(self.model, '__call__'))

        # () is link to regress
        self.assertEqual(self.model, self.model.regress)
