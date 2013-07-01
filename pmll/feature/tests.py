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
        self.assertTrue(isinstance(Feature("f", "nom").proxy, FeatureNom))
        self.assertTrue(isinstance(FeatureNom("f").proxy, FeatureNom))
        self.assertTrue(isinstance(FeatureNom("f"), FeatureNom))

    def test_feature_convert_class_lin(self):
        self.assertTrue(isinstance(
            Feature("f", scale="lin").proxy, FeatureLin))
        self.assertTrue(isinstance(Feature("f", "lin").proxy, FeatureLin))
        self.assertTrue(isinstance(FeatureLin("f").proxy, FeatureLin))
        self.assertTrue(isinstance(FeatureLin("f"), FeatureLin))

    def test_feature_convert_class_rank(self):
        self.assertTrue(isinstance(
            Feature("f", scale="rank").proxy, FeatureRank))
        self.assertTrue(isinstance(Feature("f", "rank").proxy, FeatureRank))
        self.assertTrue(isinstance(FeatureRank("f").proxy, FeatureRank))
        self.assertTrue(isinstance(FeatureRank("f"), FeatureRank))

    def test_feature_convert_class_bin(self):
        self.assertTrue(isinstance(
            Feature("f", scale="bin").proxy, FeatureBin))
        self.assertTrue(isinstance(Feature("f", "bin").proxy, FeatureBin))
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
