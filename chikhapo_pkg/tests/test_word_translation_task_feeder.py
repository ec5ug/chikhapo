import unittest
from chikhapo import TaskFeeder

class TestWordTranslationTaskFeeder(unittest.TestCase):
    def test_get_word_translation_data_for_lang_pair_less_than_300_words(self):
        feeder = TaskFeeder("word_translation")
        list_of_words = feeder.get_data_for_lang_pair("aac_eng")
        self.assertIsInstance(list_of_words, list)
        self.assertGreater(len(list_of_words), 0)
        self.assertLess(len(list_of_words), 300)
        self.assertTrue(all(isinstance(w, str) for w in list_of_words))

    def test_get_word_translation_data_for_lang_pair_more_than_300_words_lite_true(self):
        feeder = TaskFeeder("word_translation")
        list_of_words = feeder.get_data_for_lang_pair("aar_eng") # by default is lite set to True
        self.assertIsInstance(list_of_words, list)
        self.assertTrue(len(list_of_words) == 300)
        self.assertTrue(all(isinstance(w, str) for w in list_of_words))

    def test_get_word_translation_data_for_lang_pair_more_than_300_words_lite_false(self):
        feeder = TaskFeeder("word_translation")
        list_of_words = feeder.get_data_for_lang_pair("aar_eng", lite=False)
        self.assertIsInstance(list_of_words, list)
        self.assertTrue(len(list_of_words) >= 300)
        self.assertTrue(all(isinstance(w, str) for w in list_of_words))

    def test_get_word_translation_data_for_lang_pair_more_than_300_is_determistically_random(self):
        feeder = TaskFeeder("word_translation")
        list_of_words_1 = feeder.get_data_for_lang_pair("aar_eng")
        list_of_words_2 = feeder.get_data_for_lang_pair("aar_eng")
        self.assertEqual(set(list_of_words_1), set(list_of_words_2))

    def test_get_word_translation_prompts_for_lang_pair_less_than_300_words(self):
        feeder = TaskFeeder("word_translation")
        list_of_prompts = feeder.get_prompts_for_lang_pair("aya-101", "aac_eng")
        self.assertIsInstance(list_of_prompts, list)
        self.assertGreater(len(list_of_prompts), 0)
        self.assertLess(len(list_of_prompts), 300)
        self.assertTrue(all(isinstance(w, str) for w in list_of_prompts))

    def task_get_word_tanslation_prompts_for_lang_pair_more_than_300_words_lite_false(sekf):
        feeder = TaskFeeder("word_translation")
        list_of_prompts = feeder.get_prompts_for_lang_pair("aya-101", "aar_eng", lite=False)
        self.assertIsInstance(list_of_prompts, list)
        self.assertGreater(len(list_of_prompts), 300)
        self.assertTrue(all(isinstance(w, str) for w in list_of_prompts))

if __name__ == "__main__":
    unittest.main()
