import unittest
from chikhapo import TaskFeeder
import random

class TestTranslationConditionedLanguageModelingTaskFeeder(unittest.TestCase):
    def setUp(self):
        self.feeder = TaskFeeder("translation_conditioned_language_modeling")

    def test_get_data_for_lang_pair_lite_False(self):
        srcSentence_wordIndex_truncatedTrunslation_nextWord = self.feeder.get_data_for_lang_pair("spa_Latn_eng", lite=False)
        self.assertIsInstance(srcSentence_wordIndex_truncatedTrunslation_nextWord, dict)
        # checking first record
        firstKey = list(srcSentence_wordIndex_truncatedTrunslation_nextWord.keys())[0]
        self.assertIsInstance(firstKey, tuple)
        self.assertIsInstance(firstKey[0], str)  # src_sentence
        self.assertEqual(firstKey[1], 0)  # word_index
        firstValue = srcSentence_wordIndex_truncatedTrunslation_nextWord[firstKey]
        self.assertIsInstance(firstValue, dict)
        self.assertIn("truncated_translation", firstValue)
        self.assertEqual("", firstValue["truncated_translation"])
        self.assertIn("next_word", firstValue)
        self.assertIsInstance(firstValue["next_word"], str)
        self.assertNotIn(" ", firstValue["next_word"])
        # checking second record
        secondKey = list(srcSentence_wordIndex_truncatedTrunslation_nextWord.keys())[1]
        self.assertIsInstance(secondKey, tuple)
        self.assertIsInstance(secondKey[0], str)  # src_sentence
        self.assertEqual(secondKey[1], 1)  # word_index
        secondValue = srcSentence_wordIndex_truncatedTrunslation_nextWord[secondKey]
        self.assertIsInstance(secondValue, dict)
        self.assertIn("truncated_translation", secondValue)
        self.assertIsInstance(secondValue["truncated_translation"], str)
        self.assertGreater(len(secondValue["truncated_translation"]), 0)
        self.assertIn("next_word", secondValue)
        self.assertIsInstance(secondValue["next_word"], str)
        self.assertGreater(len(secondValue["next_word"]), 0)
        self.assertNotIn(" ", secondValue["next_word"])

    def test_get_data_for_lang_pair_lite_True(self):
        srcSentence_wordIndex_truncatedTrunslation_nextWord = self.feeder.get_data_for_lang_pair("spa_Latn_eng", lite=True)
        self.assertIsInstance(srcSentence_wordIndex_truncatedTrunslation_nextWord, dict)
        self.assertEqual(300, len(srcSentence_wordIndex_truncatedTrunslation_nextWord))

    def test_get_prompts_for_lang_pair_to_eng_lite_True(self):
        prompts = self.feeder.get_prompts_for_lang_pair("spa_Latn_eng", lite=True)
        self.assertEqual(300, len(prompts))
        prompt = random.choice(prompts)
        self.assertTrue(prompt.startswith("Translate the sentence into English:\nSpanish: "))
        self.assertIn("\nEnglish: ", prompt)

    def test_get_prompts_for_lang_pair_to_eng_lite_False(self):
        prompts = self.feeder.get_prompts_for_lang_pair("spa_Latn_eng", lite=False)
        self.assertGreater(len(prompts), 300)
        prompt = random.choice(prompts)
        self.assertTrue(prompt.startswith("Translate the sentence into English:\nSpanish: "))
        self.assertIn("\nEnglish: ", prompt)

    def test_get_prompts_for_lang_pair_from_eng_(self):
        prompts = self.feeder.get_prompts_for_lang_pair("eng_spa_Latn")
        self.assertEqual(len(prompts), 300)
        prompt = random.choice(prompts)
        self.assertTrue(prompt.startswith("Translate the following text into Spanish.\nEnglish: "))
        self.assertIn("\nSpanish:", prompt)

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
        