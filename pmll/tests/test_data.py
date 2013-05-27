# -*- coding: utf-8 -*-
import unittest
from ..data import (
    Feature,
    Data,
    DataReader,
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

    def test_convert(self):
        self.assertTrue(isinstance(self.feature.convert('1.0'), str))
        self.assertEqual(self.feature.convert('1.0'), '1.0')

        self.assertTrue(isinstance(self.feature_nom.convert('1.0'), str))
        self.assertEqual(self.feature.convert('1.0'), '1.0')

        self.assertTrue(isinstance(self.feature_lin.convert('1.0'), float))
        self.assertEqual(self.feature_lin.convert('1'), 1.0)

        self.assertTrue(isinstance(self.feature_rank.convert('1.0'), float))
        self.assertEqual(self.feature_rank.convert('1.0'), 1.0)

        self.assertTrue(isinstance(self.feature_bin.convert('1.0'), bool))
        self.assertEqual(self.feature_bin.convert('1.0'), True)


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

    def test_getitem(self):
        data = Data([(0, 1), (2, 3)])
        self.assertEqual(data[:], data)
        self.assertEqual(data[0, :], Data([(0, 1)]))
        self.assertEqual(data[0], Data([(0, 1)]))
        self.assertEqual(data[1], Data([(2, 3)]))
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


if __name__ == "__main__":
    unittest.main()