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
