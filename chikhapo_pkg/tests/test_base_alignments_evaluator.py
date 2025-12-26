import json
import os
import pprint

from chikhapo import Evaluator
from .base_evaluator_test import BaseEvaluatorTest

class TestAlignmentsEvaluator(BaseEvaluatorTest):
    def setUp(self):
        super().setUp()
        self.evaluator = Evaluator("translation_conditioned_language_modeling")
        # testing for reading in the test file is done in test_base_evaluator
        self.evaluator.src_lang = "spa"
        self.evaluator.tgt_lang = "eng"
        self.evaluator.data = [
            {"src_sentence": "El gato duerme en el sofa todo el dia.", "tgt_sentence_gt": "The cat sleeps on the sofa all day."},
            {"src_sentence": "Nosotros aprendemos mucho cuando trabajamos juntos.", "tgt_sentence_gt": "We learn a lot when we work together."},
            {"src_sentence": "Ella cerro la puerta antes de salir de casa.", "tgt_sentence_gt": "She closed the door before leaving the house."}
        ]
    
    def test_convert_src_tgt_sentences_to_temp_file_empty_data(self):
        self.evaluator.data = []
        self.temp_file_path = self.evaluator.convert_src_tgt_sentences_to_temp_file()
        self.assertTrue(os.path.exists(self.temp_file_path))
        with open(self.temp_file_path, "r") as f:
            content = f.read()
        self.assertEqual("", content)
    
    def test_convert_src_tgt_sentences_to_temp_file_reversal_false(self):
        self.temp_file_path = self.evaluator.convert_src_tgt_sentences_to_temp_file(reverse=False)
        self.assertTrue(os.path.exists(self.temp_file_path))
        with open(self.temp_file_path, "r") as f:
            lines = f.readlines()
        self.assertEqual(3, len(lines))
        self.assertEqual("El gato duerme en el sofa todo el dia. ||| The cat sleeps on the sofa all day.", lines[0].strip())
        self.assertEqual("Nosotros aprendemos mucho cuando trabajamos juntos. ||| We learn a lot when we work together.", lines[1].strip())
        self.assertEqual("Ella cerro la puerta antes de salir de casa. ||| She closed the door before leaving the house.", lines[2].strip())
    
    def test_convert_src_tgt_sentences_to_temp_file_reversal_true(self):
        self.temp_file_path = self.evaluator.convert_src_tgt_sentences_to_temp_file(reverse=True)
        self.assertTrue(os.path.exists(self.temp_file_path))
        with open(self.temp_file_path, "r") as f:
            lines = f.readlines()
        self.assertEqual(3, len(lines))
        self.assertEqual("The cat sleeps on the sofa all day. ||| El gato duerme en el sofa todo el dia.", lines[0].strip())
        self.assertEqual("We learn a lot when we work together. ||| Nosotros aprendemos mucho cuando trabajamos juntos.", lines[1].strip())
        self.assertEqual("She closed the door before leaving the house. ||| Ella cerro la puerta antes de salir de casa.", lines[2].strip())
    
    def test_run_fastalign_equivalence(self):
        temp_input_file = self.create_file(
            "temp_input_file.txt",
            "El gato duerme en el sofa todo el dia. ||| The cat sleeps on the sofa all day.\n" \
            "Nosotros aprendemos mucho cuando trabajamos juntos. ||| We learn a lot when we work together.\n" \
            "Ella cerro la puerta antes de salir de casa. ||| She closed the door before leaving the house."
        )
        temp_output_file = self.evaluator.run_fastalign(temp_input_file)
        with open(temp_output_file, "r") as f:
            lines = f.readlines()
        self.assertEqual(3, len(lines))
        for i, line in enumerate(lines):
            src_word_count = len(self.evaluator.data[i]["src_sentence"].split())
            tgt_word_count = len(self.evaluator.data[i]["tgt_sentence_gt"].split())
            line = line.strip()
            ints_to_ints = line.split()
            src_ints = [int(int_to_int.split("-")[0]) for int_to_int in ints_to_ints]
            tgt_ints = [int(int_to_int.split("-")[1]) for int_to_int in ints_to_ints]
            self.assertTrue(max(src_ints) <= src_word_count-1)
            self.assertTrue(max(tgt_ints) <= tgt_word_count-1)

    def test_process_fastalign_alignments_empty_data(self):
        self.evaluator.data = []
        srcWord_to_tgtWord_alignments = self.evaluator.process_fastalign_alignments(int_to_int_alignments=[])
        self.assertEqual({}, srcWord_to_tgtWord_alignments)

    def test_process_fastalign_alignments_reversal_false(self):
        srcWord_to_tgtWord_alignments = self.evaluator.process_fastalign_alignments([
            "0-0 1-1 2-2 3-3 4-4 5-5 6-6 8-7",
            "0-0 1-1 2-3 3-4 4-6 5-7",
            "0-0 1-1 3-3 4-4 6-5 8-7"
        ], reverse=False)
        self.assertEqual(3, len(srcWord_to_tgtWord_alignments))
        self.assertEqual({
            "El": {"The": 1},
            "gato": {"cat": 1},
            "duerme": {"sleeps": 1},
            "en": {"on": 1},
            "el": {"the": 1},
            "sofa": {"sofa": 1},
            "todo": {"all": 1},
            "dia.": {"day.": 1}
        }, srcWord_to_tgtWord_alignments[0])
        self.assertEqual({
            "Nosotros": {"We": 1},
            "aprendemos": {"learn": 1},
            "mucho": {"lot": 1},
            "cuando": {"when": 1},
            "trabajamos": {"work": 1},
            "juntos.": {"together.": 1}
        }, srcWord_to_tgtWord_alignments[1])
        self.assertEqual({
            "Ella": {"She": 1},
            "cerro": {"closed": 1},
            "puerta": {"door": 1},
            "antes": {"before": 1},
            "salir": {"leaving": 1},
            "casa.": {"house.": 1}
        }, srcWord_to_tgtWord_alignments[2])

    def test_process_fastalign_alignments_reversal_true(self):
        srcWord_to_tgtWord_alignments = self.evaluator.process_fastalign_alignments([
            # 0   1    2      3  4   5    6    7    8
            # The cat  sleeps on the sofa all  day.
            # El  gato duerme en el  sofa todo el   dia.
            "0-0 1-1 2-2 3-3 4-4 5-5 6-6 7-8",
            # 0        1          2     3      4          5       6    7
            # We       learn      a     lot    when       we      work together.
            # Nosotros aprendemos mucho cuando trabajamos juntos.
            "0-0 1-1 3-2 4-3 6-4 7-5",
            # 0    1      2   3      4      5       6     7      8
            # She  closed the door   before leaving the   house.
            # Ella cerro  la  puerta antes  de      salir de     casa.
            "0-0 1-1 3-3 4-4 5-6 7-8"
        ], reverse=True)
        self.assertEqual({
            "The": {"El": 1},
            "cat": {"gato": 1},
            "sleeps": {"duerme": 1},
            "on": {"en": 1},
            "the": {"el": 1},
            "sofa": {"sofa": 1},
            "all": {"todo": 1},
            "day.": {"dia.": 1}
        }, srcWord_to_tgtWord_alignments[0])
        self.assertEqual({
            "We": {"Nosotros": 1},
            "learn": {"aprendemos": 1},
            "lot": {"mucho": 1},
            "when": {"cuando": 1},
            "work": {"trabajamos": 1},
            "together.": {"juntos.": 1}
        }, srcWord_to_tgtWord_alignments[1])
        self.assertEqual({
            "She": {"Ella": 1},
            "closed": {"cerro": 1},
            "door": {"puerta": 1},
            "before": {"antes": 1},
            "leaving": {"salir": 1},
            "house.": {"casa.": 1}
        }, srcWord_to_tgtWord_alignments[2])

    def test_process_fastalign_alignments_int_to_int_alignments_below_0(self):
        with self.assertRaises(Exception) as ctx:
            self.evaluator.process_fastalign_alignments([
                "0-0 1-1 2-2 3-3 4-4 5-5 -6-6 7-8",
                "0-0 1-1 3-2 4-3 6-4 7-5",
                "0-0 1-1 3-3 4-4 5-6 7-8"
            ])
        self.assertEqual("The alignment -6-6 should be demarcated with a - and should not include negatives.", str(ctx.exception))

    def test_process_fastalign_alignments_src_int_greater_than_word_count(self):
        with self.assertRaises(Exception) as ctx:
            self.evaluator.process_fastalign_alignments([
                "0-0 1-1 2-2 3-3 4-4 5-5 6-6 100-8",
                "0-0 1-1 3-2 4-3 6-4 7-5",
                "0-0 1-1 3-3 4-4 5-6 7-8"
            ])
        self.assertEqual("Source alignment 100-8 should be within the number of words in the source sentence.", str(ctx.exception))

    def test_process_fastalign_alignments_tgt_int_greater_than_word_count(self):
        with self.assertRaises(Exception) as ctx:
            self.evaluator.process_fastalign_alignments([
                "0-0 1-1 2-2 3-3 4-4 5-5 6-6 7-100",
                "0-0 1-1 3-2 4-3 6-4 7-5",
                "0-0 1-1 3-3 4-4 5-6 7-8"
            ])
        self.assertEqual("Target alignment 7-100 should be within the number of words in the target sentence.", str(ctx.exception))

    def test_get_statistical_alignments_empty_data(self):
        self.evaluator.data = []
        word_to_word_alignments = self.evaluator.get_statistical_alignments()
        self.assertEqual({}, word_to_word_alignments)

    def test_process_fastalign_alignments_reversal_false(self):
        word_to_word_alignments = self.evaluator.get_statistical_alignments(reverse=False)
        self.assertIsInstance(word_to_word_alignments, dict)
        self.assertEqual(3, len(word_to_word_alignments))
        for i in range(len(word_to_word_alignments)):
            words_in_src_sentence = self.evaluator.data[i]["src_sentence"].split()
            words_in_tgt_sentence_gt = self.evaluator.data[i]["tgt_sentence_gt"].split()
            for src_word in word_to_word_alignments[i].keys():
                self.assertIn(src_word, words_in_src_sentence)
                tgt_words_gt = word_to_word_alignments[i][src_word]
                for tgt_word_gt in tgt_words_gt:
                    self.assertIn(tgt_word_gt, words_in_tgt_sentence_gt)
                    self.assertTrue(word_to_word_alignments[i][src_word][tgt_word_gt] > 0)

    def test_process_fastalign_alignments_reversal_true(self):
        word_to_word_alignments = self.evaluator.get_statistical_alignments(reverse=True)
        self.assertEqual(3, len(word_to_word_alignments))
        self.assertEqual(3, len(word_to_word_alignments))
        for i in range(len(word_to_word_alignments)):
            words_in_src_sentence = self.evaluator.data[i]["tgt_sentence_gt"].split()
            words_in_tgt_sentence_gt = self.evaluator.data[i]["src_sentence"].split()
            for src_word in word_to_word_alignments[i].keys():
                self.assertIn(src_word, words_in_src_sentence)
                tgt_words_gt = word_to_word_alignments[i][src_word]
                for tgt_word_gt in tgt_words_gt:
                    self.assertIn(tgt_word_gt, words_in_tgt_sentence_gt)
                    self.assertTrue(word_to_word_alignments[i][src_word][tgt_word_gt] > 0)
    