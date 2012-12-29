# -*- coding: utf-8 -*-
import unittest


__author__  = "Kirill Pavlov"
__email__   = "kirill.pavlov@phystech.edu"


class Feature(object):
    """
    Feture representation, converts feature value according to one of the
    following types:

    nom: nominal value represented by string
    lin: float number in linear scale
    rank: float number, arithmetic operations are not supported
    bin: binary format, true/false or 1/0
    """
    FEATURE_TYPE_MAPPER = {
        'nom': str,
        'lin': float,
        'rank': float,
        'bin': bool,
        }

    def __init__(self, title, scale=None):
        self.title = unicode(title)
        self.scale = scale
        self.convert = self.FEATURE_TYPE_MAPPER.get(self.scale, str)

    def __str__(self):
        return unicode(self).encode('utf8')

    def __unicode__(self):
        return "%s:%s" % (unicode(self.title), unicode(self.type))


class Data(object):
    """
    Data is general data representation. It is object x feature matrix.
    There is no label, all of the features are equal. It is job for data
    manager to define what is label.
    """
    def __init__(self, objects, features=None):
        self.features = []


class DataReader(object):
    """
    Read data form tab separated file stream either into objects or matrix
    stream can be open(file) or line generator
    """
    def __parse_header(self, header):
        heared_prefix = "# "
        if not header.startswith(heared_prefix):
            msg = 'Bad header format. Should starts from "%s"' % heared_prefix
            raise ValueError(msg)
        else:
            header = header[len(heared_prefix):].split('#', 1)[0].rstrip()
            header = header.replace(",", "\t").replace(";", "\t")

        self.features = [
            Feature(*field.split(':'))
            for field in header.split("\t")
            ]

        return self.features


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
