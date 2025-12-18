import unittest
from chikhapo import TaskFeeder

class BagOfWordsMachineTranslationFeeder(unittest.TestCase):
    def setUp(self):
        self.feeder = TaskFeeder("bag_of_words_machine_translation")

    def test_get_lang_pairs(self):
        lang_pairs = self.feeder.get_lang_pairs()
        self.assertIn("spa_Latn_eng", lang_pairs)
        self.assertIn("eng_spa_Latn", lang_pairs)
        self.assertNotIn("eng_eng_Latn", lang_pairs)
        self.assertNotIn("eng_Latn_eng", lang_pairs)
        self.assertGreaterEqual(452, len(lang_pairs)) # as of Dec 18, 2025
    
    def test_get_lang_pairs_to_eng(self):
        lang_pairs = self.feeder.get_lang_pairs(DIRECTION="X_to_eng")
        self.assertIn("spa_Latn_eng", lang_pairs)
        self.assertNotIn("eng_spa_Latn", lang_pairs)
        self.assertNotIn("eng_eng_Latn", lang_pairs)
        self.assertNotIn("eng_Latn_eng", lang_pairs)
        self.assertGreaterEqual(226, len(lang_pairs)) # as of Dec 18, 2025
    
    def test_get_lang_pairs_from_eng(self):
        lang_pairs = self.feeder.get_lang_pairs(DIRECTION="eng_to_X")
        self.assertIn("eng_spa_Latn", lang_pairs)
        self.assertNotIn("spa_Latn_eng", lang_pairs)
        self.assertNotIn("eng_eng_Latn", lang_pairs)
        self.assertNotIn("eng_Latn_eng", lang_pairs)
        self.assertGreaterEqual(226, len(lang_pairs)) # as of Dec 18, 2025

    def test_get_lang_pairs_invalid_direction(self):
        with self.assertRaises(Exception) as context:
            self.feeder.get_lang_pairs(DIRECTION="invalid_direction")
        self.assertEqual(str(context.exception), "An invalid directon was specified. It should be None, \"X_to_eng\", or \"eng_to_X\"")