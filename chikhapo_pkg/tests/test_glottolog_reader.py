import unittest
from chikhapo import GlottologReader

class TestGlottologReader(unittest.TestCase):
    def setUp(self):
        self.glottolog_reader = GlottologReader()
    
    def test_get_lang_info_empty_str(self):
        with self.assertRaisesRegex(Exception, "Please enter a valid ISO code"):
            self.glottolog_reader.get_lang_info("")

    def test_get_lang_info_invalid_iso(self):
        with self.assertRaisesRegex(Exception, "The iso aaj could not be found in the Glottolog data."):
            self.glottolog_reader.get_lang_info("aaj")

    def test_get_lang_info(self):
        record = self.glottolog_reader.get_lang_info("spa")
        self.assertEqual(1, len(record))
        dict_inst = record[0]
        self.assertEqual("spa", dict_inst["iso"])
        self.assertEqual("Spanish", dict_inst["name"])
        self.assertEqual(40.4414, dict_inst["latitude"])
        self.assertEqual(-1.11788, dict_inst["longitude"])
        expected = ["Andorra", "Argentina", "Bolivia", "Brazil", "Belize", "Chile", "Colombia", "Costa Rica", "Cuba", "Dominican Republic", "Ecuador", "Spain", "France", "Gibraltar", "Guatemala", "Guyana", "Honduras", "Haiti", "Morocco", "Mexico", "Nicaragua", "Panama", "Peru", "Puerto Rico", "Portugal", "Paraguay", "El Salvador", "United States", "Uruguay", "Venezuela"]
        actual = dict_inst["country"]
        for actual_country in actual:
            self.assertTrue(any(expected_country in actual_country or actual_country in expected_country for expected_country in expected))

    def test_get_language_to_family_dict(self):
        actual = self.glottolog_reader.get_language_to_family_dict()
        expected = {
            "spa": "Indo-European", # spanish
            "eng": "Indo-European", # english
            "deu": "Indo-European", # german
            "fra": "Indo-European", # french
            "arb": "Afro-Asiatic", # arabic
            "amh": "Afro-Asiatic", # amharic
            "heb": "Afro-Asiatic", # hebrew
            "swh": "Atlantic-Congo", # swahili
            "wol": "Atlantic-Congo", # wolof
            "fuc": "Atlantic-Congo", # pulaar
            "cmn": "Sino-Tibetan", # mandarin chinese
            "ind": "Austronesian", # indonesian
            "fin": "Uralic", # finnish
            "tur": "Turkic", # turkish
            "tam": "Dravidian", # tamil
        }
        for lang, family in expected.items():
            self.assertEqual(family, actual[lang])
    