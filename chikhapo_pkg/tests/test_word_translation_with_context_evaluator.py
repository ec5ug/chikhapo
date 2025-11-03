import unittest
from chikhapo import TaskFeeder

class TestWordTranslationWithContextTaskFeeder(unittest.TestCase):
    def test_get_word_translation_with_context_data_for_lang_pair_more_than_300_word_sentence_lite_true(self):
        feeder = TaskFeeder("word_translation_with_context")
        list_of_words_sentences = feeder.get_data_for_lang_pair("avn_Latn_eng")
        self.assertIsInstance(list_of_words_sentences, list)
        self.assertEqual(len(list_of_words_sentences), 300)
        self.assertTrue(all(isinstance(w, tuple) for w in list_of_words_sentences))
        self.assertEqual(len(list_of_words_sentences[0]), 2)

    def test_get_word_translation_with_context_data_for_lang_pair_more_than_300_word_sentence_lite_false(self):
        feeder = TaskFeeder("word_translation_with_context")
        list_of_words_sentences = feeder.get_data_for_lang_pair("avn_Latn_eng", lite=False)
        self.assertIsInstance(list_of_words_sentences, list)
        self.assertGreater(len(list_of_words_sentences), 300)
        self.assertTrue(all(isinstance(w, tuple) for w in list_of_words_sentences))
        self.assertEqual(len(list_of_words_sentences[0]), 2)

    def test_get_word_translation_with_context_prompts_for_lang_pair_more_than_300_word_sentence_lite_true(self):
        feeder = TaskFeeder("word_translation_with_context")
        list_of_prompts = feeder.get_prompts_for_lang_pair("aya-101", "avn_Latn_eng")
        self.assertIsInstance(list_of_prompts, list)
        self.assertEqual(len(list_of_prompts), 300)
        self.assertTrue(all(isinstance(w, str) for w in list_of_prompts))

    def test_get_word_translation_with_context_prompts_for_lang_pair_more_than_300_word_sentence_lite_false(self):
        feeder = TaskFeeder("word_translation_with_context")
        list_of_prompts = feeder.get_prompts_for_lang_pair("aya-101", "avn_Latn_eng")
        self.assertIsInstance(list_of_prompts, list)
        self.assertEqual(len(list_of_prompts), 300)
        self.assertTrue(all(isinstance(w, str) for w in list_of_prompts))

if __name__ == "__main__":
    unittest.main()