# -*- coding: utf-8 -*-
from collections import namedtuple
import unittest

from ..data import (
    Data,
    DataReader,
    Feature,
    FeatureBin,
    FeatureLin,
    FeatureNom,
    FeatureRank,
)

__author__ = "Kirill Pavlov"
__email__ = "kirill.pavlov@phystech.edu"


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

    def test_feature_convert_class(self):
        self.assertTrue(isinstance(Feature("f"), Feature))
        self.assertTrue(isinstance(Feature("f").proxy, FeatureNom))

        self.assertTrue(isinstance(
            Feature("f", scale="nom").proxy, FeatureNom))
        self.assertTrue(isinstance(Feature("f", "nom").proxy, FeatureNom))
        self.assertTrue(isinstance(FeatureNom("f").proxy, FeatureNom))
        self.assertTrue(isinstance(FeatureNom("f"), FeatureNom))

        self.assertTrue(isinstance(
            Feature("f", scale="lin").proxy, FeatureLin))
        self.assertTrue(isinstance(Feature("f", "lin").proxy, FeatureLin))
        self.assertTrue(isinstance(FeatureLin("f").proxy, FeatureLin))
        self.assertTrue(isinstance(FeatureLin("f"), FeatureLin))

        self.assertTrue(isinstance(
            Feature("f", scale="rank").proxy, FeatureRank))
        self.assertTrue(isinstance(Feature("f", "rank").proxy, FeatureRank))
        self.assertTrue(isinstance(FeatureRank("f").proxy, FeatureRank))
        self.assertTrue(isinstance(FeatureRank("f"), FeatureRank))

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
        self.assertEqual(Feature("f1")(Object()), "value")

        cls = namedtuple("Object", ["f1", "f2"])
        self.assertEqual(FeatureNom("f1")(cls(0, 1)), "0")
        self.assertEqual(FeatureNom("f2")(cls(0, 1)), "1")

    def test__call__data(self):
        data = Data([[1, 2, 3],
                     [3, 4, 5]])
        self.assertEqual(data.features[0](data), data[:, 0])
        self.assertEqual(data.features[1](data), data[:, 1])
        self.assertEqual(data.features[2](data), data[:, 2])


class FeatureBinTest(unittest.TestCase):
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

    def test__or__(self):
        f = self.f1 | self.f2
        self.assertEqual(f(self.Object(0, 0)), 0)
        self.assertEqual(f(self.Object(0, 1)), 1)
        self.assertEqual(f(self.Object(1, 0)), 1)
        self.assertEqual(f(self.Object(1, 1)), 1)

    def test__xor__(self):
        f = self.f1 ^ self.f2
        self.assertEqual(f(self.Object(0, 0)), 0)
        self.assertEqual(f(self.Object(0, 1)), 1)
        self.assertEqual(f(self.Object(1, 0)), 1)
        self.assertEqual(f(self.Object(1, 1)), 0)


class DataTest(unittest.TestCase):
    def setUp(self):
        self.data_file_content = "\n".join(
            [
                "# label:nom\tweight:lin\theigth:lin",
                "0\t70\t100.0",
                "1\t50\t200",
            ])
        objects, features = DataReader.read(self.data_file_content.split("\n"))
        self.data = Data(objects, features)

    def test_init(self):
        pass

    def test_init_features(self):
        nfeatures = len(self.data_file_content.split("\n", 1)[0].split("\t"))
        self.assertEqual(
            len(self.data.features),
            nfeatures,
            "Error in parsing, check data.features length != %s" % nfeatures,
        )

        for feature in self.data.features:
            self.assertIsInstance(feature, Feature)

    def test__init__similar_features(self):
        with self.assertRaises(ValueError):
            Data([[0, 0]], [Feature("f0"), Feature("f0")])

    def test__eq__(self):
        objects1 = [(0, 1)]
        objects2 = [(1, 0)]
        features1 = [Feature("f1"), Feature("f2")]
        features2 = [Feature("f2"), Feature("f1")]
        #import ipdb; ipdb.set_trace()
        self.assertEqual(Data(objects1), Data(objects1))
        self.assertEqual(Data(objects1, features1), Data(objects1, features1))
        self.assertEqual(Data(objects1, features1), Data(objects2, features2))

        self.assertNotEqual(
            Data(objects1, features1), Data(objects1, features2))

    def test_getitem_one(self):
        self.assertEqual(Data([[0]])[0, 0], (0, ))

    def test_getitem(self):
        data = Data([(0, 1), (2, 3)])
        self.assertEqual(data[:], data)
        self.assertEqual(data[0, :], (0, 1))
        self.assertEqual(data[0], (0, 1))
        self.assertEqual(data[1], (2, 3))
        self.assertEqual(data[:, 0], Data([(0, ), (2, )], [data.features[0]]))
        self.assertEqual(data[:, 1], Data([(1, ), (3, )], [data.features[1]]))

    def test_getitem_many(self):
        data = Data([(0, 1, 2)])
        self.assertEqual(data[:, 0:1], Data([(0, 1)], data.features[0:1]))
        self.assertEqual(data[:, :], data)

    def test_getitem_feature(self):
        data = Data([(0, 1)], [Feature("f1"), Feature("f2")])
        self.assertEqual(
            data[:, Feature("f1")], Data([(0, )], [Feature("f1")]))
        self.assertEqual(
            data[:, Feature("f2")], Data([(1, )], [Feature("f2")]))

    def test__add__different_number_objects(self):
        with self.assertRaises(ValueError):
            Data([[0], [1]], [Feature("f1")]) + Data([[1]], [Feature("f2")])

    def test__add__features_intersected(self):
        with self.assertRaises(ValueError):
            Data([(0, )]) + Data([(1, )])

    def test__add__(self):
        data1 = Data([[0]], [Feature("f1")]) + Data([[1]], [Feature("f2")])
        data2 = Data([[1]], [Feature("f2")]) + Data([[0]], [Feature("f1")])
        self.assertEqual(data1, data2, "Not commutative operation")
        self.assertEqual(data1, Data([[0, 1]], [Feature("f1"), Feature("f2")]))

    def test_vif_one_feature(self):
        with self.assertRaises(ValueError):
            Data([(0, )]).vif
            Data([(0, ), (1, )]).vif

    def test_vif_feateres_more_than_objects(self):
        with self.assertRaises(ValueError):
            Data([(0, 1)]).vif
            Data([(0, 1, 2), (1, 2, 3)]).vif

    def test_vif(self):
        self.assertEqual(Data([(0, 1), (1, 2)]).vif, [1.25, 0.25])


class DataReaderTest(unittest.TestCase):
    def setUp(self):
        self.header = "# label:nom\tweight:lin\theigth:lin"
        self.data_file_content = "\n".join(
            [
                "# label:nom\tweight:lin\theigth:lin",
                "0\t70\t100.0",
                "1\t50\t200",
            ])

    def test_parse_header(self):
        DataReader._DataReader__parse_header(self.header)

    def test_init_bad_header_hash(self):
        header = self.header[1:]
        with self.assertRaises(ValueError):
            DataReader._DataReader__parse_header(header)

    def test_init_duplicated_features(self):
        last_feature = self.header.rsplit("\t", 1)[-1]
        header = "\t".join([self.header, last_feature])
        with self.assertRaises(ValueError):
            DataReader._DataReader__parse_header(header)

    def test_init_skipped_feature_type(self):
        header = self.header.rsplit(":", 1)[0]
        features = DataReader._DataReader__parse_header(header)
        self.assertEqual(features[-1].scale, Feature.DEFAULT_SCALE)

    def test_get_duplicated_features(self):
        features = [Feature("f", "lin"), Feature("f")]
        duplicated_features =\
            DataReader._DataReader__get_duplicated_features(features)
        self.assertTrue("f" in duplicated_features)

    def test_read(self):
        objects, features = DataReader.read(self.data_file_content.split("\n"))
        self.assertEqual(len(features), 3)
        for feature in features:
            self.assertIsInstance(feature, Feature)

        objects_list = list(objects)
        self.assertEqual(len(objects_list), 2)
