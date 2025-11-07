import unittest
from chikhapo import TaskFeeder

class TestWordTranslationTaskFeeder(unittest.TestCase):
    def setUp(self):
        self.feeder = TaskFeeder("word_translation")

    def test_get_word_translation_data_for_lang_pair_less_than_300_words(self):
        list_of_words = self.feeder.get_data_for_lang_pair("aac_eng")
        # self.assertIsInstance(list_of_words, list)
        self.assertGreater(len(list_of_words), 0)
        self.assertLess(len(list_of_words), 300)
        self.assertTrue(all(isinstance(w, str) for w in list_of_words))

    def test_get_word_translation_data_for_lang_pair_more_than_300_words_lite_true(self):
        list_of_words = self.feeder.get_data_for_lang_pair("aar_eng") # by default is lite set to True
        # self.assertIsInstance(list_of_words, list)
        self.assertTrue(len(list_of_words) == 300)
        self.assertTrue(all(isinstance(w, str) for w in list_of_words))

    def test_get_word_translation_data_for_lang_pair_more_than_300_words_lite_false(self):
        list_of_words = self.feeder.get_data_for_lang_pair("aar_eng", lite=False)
        # self.assertIsInstance(list_of_words, list)
        self.assertTrue(len(list_of_words) >= 300)
        self.assertTrue(all(isinstance(w, str) for w in list_of_words))

    def test_get_word_translation_data_for_lang_pair_more_than_300_is_determistically_random(self):
        words_1 = self.feeder.get_data_for_lang_pair("aar_eng")
        words_2 = self.feeder.get_data_for_lang_pair("aar_eng")
        self.assertEqual(words_1, words_2)

    def test_get_word_translation_prompts_for_lang_pair_less_than_300_words(self):
        list_of_prompts = self.feeder.get_prompts_for_lang_pair("aac_eng")
        self.assertIsInstance(list_of_prompts, list)
        self.assertGreater(len(list_of_prompts), 0)
        self.assertLess(len(list_of_prompts), 300)
        self.assertTrue(all(isinstance(w, str) for w in list_of_prompts))

    def test_get_word_tanslation_prompts_for_lang_pair_more_than_300_words_lite_false(self):
        list_of_prompts = self.feeder.get_prompts_for_lang_pair("aar_eng", lite=False)
        self.assertIsInstance(list_of_prompts, list)
        self.assertGreater(len(list_of_prompts), 300)
        self.assertTrue(all(isinstance(w, str) for w in list_of_prompts))

    def test_get_lang_pairs(self):
        lang_pairs = self.feeder.get_lang_pairs()
        self.assertIn("spa_eng", lang_pairs)
        self.assertIn("eng_spa", lang_pairs)
        self.assertNotIn("all_eng", lang_pairs)
        self.assertNotIn("eng_all", lang_pairs)

if __name__ == "__main__":
    unittest.main()
