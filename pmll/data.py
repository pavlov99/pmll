# -*- coding: utf-8 -*-
import unittest


class Feature(object):
    pass


class Data(object):
    def __init__(self, stream):
        self.features = []


class FeatureTest(unittest.TestCase):
    pass


class DataTest(unittest.TestCase):
    def setUp(self):
        self.data_file_content = "\n".join([
                "# label:nom\tweight:lin\theigth:lin",
                "0\t70\t100.0",
                "1\t50\t200",
                ])
        self.data = Data(self.data_file_content.split("\n"))

    def test_init(self):
        pass

    def test_init_bad_header_hash(self):
        data_file_content = self.data_file_content[1:]
        with self.assertRaises(ValueError):
            Data(data_file_content.split("\n")) 

    def test_init_duplicated_features(self):
        header = self.data_file_content.split("\n")[0]
        last_feature = header.rsplit("\t", 1)[-1]
        data_file_content = "\n".join([
                "\t".join([header, last_feature])] +\
                self.data_file_content.split("\n")[1:],
                )

        with self.assertRaises(ValueError):
            Data(data_file_content.split("\n")) 

    def test_init_skipped_feature_type(self):
        header, body = self.data_file_content.split("\n", 1)
        header = header.rsplit(":", 1)[0]
        data_file_content = "\n".join([header, body])
        Data(data_file_content.split("\n"))

    def test_init_features(self):
        nfeatures = len(self.data_file_content.split("\n", 1)[0].split("\t"))
        self.assertEqual(
            len(self.data.features),
            nfeatures,
            "Error in parsing, check data.features length != %s" % nfeatures,
            )

        for feature in self.data.features:
            self.assertIsInstance(feature, Feature)


if __name__ == "__main__":
    unittest.main()
