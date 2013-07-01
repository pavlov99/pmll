# -*- coding: utf-8 -*-
from collections import namedtuple
import unittest

from . import (
    Feature,
    FeatureBin,
    FeatureLin,
    FeatureNom,
    FeatureRank,
)


class FeatureTest(unittest.TestCase):

    """Test Feature class in general"""

    def setUp(self):
        self.feature = Feature('f')
        self.feature_nom = Feature('f', 'nom')
        self.feature_lin = Feature('f', 'lin')
        self.feature_rank = Feature('f', 'rank')
        self.feature_bin = Feature('f', 'bin')
        self.feature_nom2 = Feature('f', 'nom')

    def test__eq__(self):
        self.assertNotEqual(id(self.feature_nom), id(self.feature_nom2))
        self.assertEqual(self.feature_nom, self.feature_nom2)
        self.assertNotEqual(self.feature_nom, self.feature_lin)

    def test_feature_convert_class_default(self):
        self.assertTrue(isinstance(Feature("f"), Feature))
        self.assertTrue(isinstance(Feature("f").proxy, FeatureLin))

    def test_feature_convert_class_nom(self):
        self.assertTrue(isinstance(
            Feature("f", scale="nom").proxy, FeatureNom))
        self.assertTrue(isinstance(FeatureNom("f").proxy, FeatureNom))
        self.assertTrue(isinstance(FeatureNom("f"), FeatureNom))

    def test_feature_convert_class_lin(self):
        self.assertTrue(isinstance(
            Feature("f", scale="lin").proxy, FeatureLin))
        self.assertTrue(isinstance(FeatureLin("f").proxy, FeatureLin))
        self.assertTrue(isinstance(FeatureLin("f"), FeatureLin))

    def test_feature_convert_class_rank(self):
        self.assertTrue(isinstance(
            Feature("f", scale="rank").proxy, FeatureRank))
        self.assertTrue(isinstance(FeatureRank("f").proxy, FeatureRank))
        self.assertTrue(isinstance(FeatureRank("f"), FeatureRank))

    def test_feature_convert_class_bin(self):
        self.assertTrue(isinstance(
            Feature("f", scale="bin").proxy, FeatureBin))
        self.assertTrue(isinstance(FeatureBin("f").proxy, FeatureBin))
        self.assertTrue(isinstance(FeatureBin("f"), FeatureBin))

    def test_feature_scales(self):
        self.assertEqual(FeatureBin.scale, "bin")
        self.assertEqual(FeatureLin.scale, "lin")
        self.assertEqual(FeatureNom.scale, "nom")
        self.assertEqual(FeatureRank.scale, "rank")

    def test_feature_convert(self):
        self.assertEqual(FeatureNom.convert('1.0'), '1.0')
        self.assertEqual(FeatureLin.convert('1'), 1.0)
        self.assertEqual(FeatureRank.convert('1.0'), 1)
        self.assertEqual(FeatureBin.convert('1.0'), True)
        self.assertEqual(FeatureBin.convert(''), False)

    def test__call__object(self):
        class Object(object):
            f1 = "value"
        self.assertEqual(FeatureNom("f1")(Object()), "value")

        cls = namedtuple("Object", ["f1", "f2"])
        self.assertEqual(FeatureNom("f1")(cls(0, 1)), "0")
        self.assertEqual(FeatureNom("f2")(cls(0, 1)), "1")


class FeatureBinTest(unittest.TestCase):

    """Test Binary Feature class functionality"""

    def setUp(self):
        self.f1 = FeatureBin("f1")
        self.f2 = FeatureBin("f2")
        self.Object = namedtuple("Object", ["f1", "f2"])

    def test__and__(self):
        f = self.f1 & self.f2
        self.assertEqual(f(self.Object(0, 0)), 0)
        self.assertEqual(f(self.Object(0, 1)), 0)
        self.assertEqual(f(self.Object(1, 0)), 0)
        self.assertEqual(f(self.Object(1, 1)), 1)

    def test__and__constant(self):
        f = self.f1 & True
        self.assertEqual(f.formula, self.f1.formula)
        f = self.f1 & False
        self.assertEqual(f.formula, False)

    def test__rand__constant(self):
        f = True & self.f1
        self.assertEqual(f.formula, self.f1.formula)
        f = False & self.f1
        self.assertEqual(f.formula, False)

    def test__or__(self):
        f = self.f1 | self.f2
        self.assertEqual(f(self.Object(0, 0)), 0)
        self.assertEqual(f(self.Object(0, 1)), 1)
        self.assertEqual(f(self.Object(1, 0)), 1)
        self.assertEqual(f(self.Object(1, 1)), 1)

    def test__or__constant(self):
        f = self.f1 | True
        self.assertEqual(f.formula, True)
        f = self.f1 | False
        self.assertEqual(f.formula, self.f1.formula)

    def test__ror__constant(self):
        f = True | self.f1
        self.assertEqual(f.formula, True)
        f = False | self.f1
        self.assertEqual(f.formula, self.f1.formula)

    def test__xor__(self):
        f = self.f1 ^ self.f2
        self.assertEqual(f(self.Object(0, 0)), 0)
        self.assertEqual(f(self.Object(0, 1)), 1)
        self.assertEqual(f(self.Object(1, 0)), 1)
        self.assertEqual(f(self.Object(1, 1)), 0)

    def test__xor__constant(self):
        f = self.f1 ^ True
        self.assertEqual(f(self.Object(0, 0)), True)
        self.assertEqual(f(self.Object(1, 0)), False)
        f = self.f1 ^ False
        self.assertEqual(f.formula, self.f1.formula)

    def test__rxor__constant(self):
        f = True ^ self.f1
        self.assertEqual(f(self.Object(0, 0)), True)
        self.assertEqual(f(self.Object(1, 0)), False)
        f = False ^ self.f1
        self.assertEqual(f.formula, self.f1.formula)

    def test_complex(self):
        g = self.f1 | self.f2
        f = g & self.f2
        self.assertEqual(f(self.Object(0, 0)), 0)
        self.assertEqual(f(self.Object(0, 1)), 1)
        self.assertEqual(f(self.Object(1, 0)), 0)
        self.assertEqual(f(self.Object(1, 1)), 1)


class FeatureLinTest(unittest.TestCase):

    """Test Linear Feature class functionality"""

    def setUp(self):
        self.f1 = FeatureLin("f1")
        self.f2 = FeatureLin("f2")
        self.f3 = FeatureLin("f3")
        self.Object = namedtuple("Object", ["f1", "f2", "f3"])

    def test__neg__(self):
        f = -self.f1
        self.assertEqual(f(self.Object(1, 0, 0)), -1)

    def test__add__(self):
        f = self.f1 + self.f2
        self.assertEqual(f(self.Object(2, 3, 7)), 5)

        f = self.f1 + self.f2 + self.f3
        self.assertEqual(f(self.Object(2, 3, 7)), 12)

    def test__add__constant(self):
        f = self.f1 + 1
        self.assertEqual(f(self.Object(2, 0, 0)), 3)

    def test__radd__constant(self):
        f = 1 + self.f1
        self.assertEqual(f(self.Object(2, 0, 0)), 3)

    def test__sub__(self):
        f = self.f1 - self.f2
        self.assertEqual(f(self.Object(2, 3, 7)), -1)

        f = self.f1 - self.f2 - self.f3
        self.assertEqual(f(self.Object(7, 3, 4)), 0)

    def test__sub__constant(self):
        f = self.f1 - 1
        self.assertEqual(f(self.Object(3, 0, 0)), 2)

    def test__rsub__constant(self):
        f = 1 - self.f1
        self.assertEqual(f(self.Object(3, 0, 0)), -2)

    def test__mul__(self):
        f = self.f1 * self.f2
        self.assertEqual(f(self.Object(2, 3, 7)), 6)

        f = self.f1 * self.f2 * self.f3
        self.assertEqual(f(self.Object(2, 3, 7)), 42)

    def test__mul__constant(self):
        f = self.f1 * 3
        self.assertEqual(f(self.Object(2, 0, 0)), 6)

    def test__rmul__constant(self):
        f = 3 * self.f1
        self.assertEqual(f(self.Object(2, 0, 0)), 6)

    def test__div__(self):
        f = self.f1 / self.f2
        self.assertEqual(f(self.Object(3, 2, 7)), 1.5)

        f = self.f1 / self.f2 / self.f3
        self.assertEqual(f(self.Object(30, 3, 2)), 5)

    def test__div__constant(self):
        f = self.f1 / 3
        self.assertEqual(f(self.Object(6, 0, 0)), 2)

    def test__rdiv__constant(self):
        f = 3 / self.f1
        self.assertEqual(f(self.Object(6, 0, 0)), 0.5)

    def test__pow__(self):
        f = self.f1 ** self.f2
        self.assertEqual(f(self.Object(2, 3, 7)), 8)

    def test__pow__constant(self):
        f = self.f1 ** 2
        self.assertEqual(f(self.Object(3, 0, 0)), 9)

    def test__rpow__constant(self):
        f = 2 ** self.f1
        self.assertEqual(f(self.Object(3, 0, 0)), 8)
