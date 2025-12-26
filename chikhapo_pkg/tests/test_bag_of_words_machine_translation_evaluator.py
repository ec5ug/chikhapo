import json
import os
import tempfile
import unittest
from unittest.mock import patch
import pprint

from .base_evaluator_test import BaseEvaluatorTest
from chikhapo import Evaluator

class TestTranslationConditionedLanguageModelingEvaluator(BaseEvaluatorTest):
    def setUp(self):
        super().setUp()
        self.evaluator = Evaluator("bag_of_words_machine_translation")
    
    def test_no_src_sentence(self):
        path = self.create_file(
            "missing_src_sentence.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"tgt_sentence_gt": "a", "tgt_sentence_pred": "a"}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn("The key \"src_sentence\" is not found in ", str(ctx.exception))
    
    def test_improper_src_sentence(self):
        path = self.create_file(
            "improper_src_sentence.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": 1, "tgt_sentence_gt": "a", "tgt_sentence_pred": "a"}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn(": The source sentence in \"src_sentence\" should be stored as a string, not ", str(ctx.exception))

    def test_no_tgt_sentence_gt(self):
        path = self.create_file(
            "missing_tgt_sentence_gt.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": "a", "tgt_sentence_pred": "a"}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn("The key \"tgt_sentence_gt\" is not found in ", str(ctx.exception))

    def test_improper_tgt_sentence_gt(self):
        path = self.create_file(
            "improper_src_sentence.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": "a", "tgt_sentence_gt": 1, "tgt_sentence_pred": "a"}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn(": The ground-truth target sentence in \"tgt_sentence_gt\" should be stored as a string, not ", str(ctx.exception))
    
    def test_no_tgt_sentence_pred(self):
        path = self.create_file(
            "improper_tgt_sentence_pred.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": "a", "tgt_sentence_gt": "a"}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn("The key \"tgt_sentence_pred\" is not found in ", str(ctx.exception))

    def test_improper_tgt_sentence_pred(self):
        path = self.create_file(
            "improper_tgt_sentence_pred.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": "a", "tgt_sentence_gt": "a", "tgt_sentence_pred": 1}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn(": The predicted target sentence in \"tgt_sentence_pred\" should be stored as a string, not ", str(ctx.exception))

    def test_X_to_eng_incomplete_lexicons_and_alignments(self):
        path = self.create_file(
            "X_to_eng_incomplete_lexicons_and_alignments.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{
                    "src_sentence": "Mi hermano como manzanas rojas.",
                    "tgt_sentence_gt": "My brother eats red apples.",
                    "tgt_sentence_pred": "My brother eats red apples."
                }]
            })
        )
        fake_lexicon = [
            {"source_word": "hermano", "target_translations": ["brother"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "como", "target_translations": ["eats"], "src_lang": "spa", "tgt_lang": "eng"}
        ]
        fake_word_to_word_alignments = {
            0: {
                "hermano": {"brother": 1},
                "manzanas": {"apples": 1}
            }
        }
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(3, len(self.evaluator.xword_class_pred))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["hermano"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["hermano"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["hermano"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["como"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["como"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["como"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["manzanas"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["manzanas"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["manzanas"])
        self.assertEqual(100, self.evaluator.lang_score)
    
    def test_X_to_eng_using_only_alignments_exact_match(self):
        path = self.create_file(
            "X_to_eng_using_only_alignments_exact_match.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{
                    "src_sentence": "El gato negro duerme en el sofá.",
                    "tgt_sentence_gt": "The black cat sleeps on the couch.",
                    "tgt_sentence_pred": "The black cat sleeps on the couch."
                }]
            })
        )
        fake_lexicon = [
            {"source_word": "casa", "target_translations": ["house"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "banco", "target_translations": ["bank", "bench"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "correr", "target_translations": ["to run"], "src_lang": "spa", "tgt_lang": "eng"}
        ]
        fake_word_to_word_alignments = {
            0: {
                "El":{"The":3, "black":2},
                "gato":{"cat":2, "couch":1},
                "negro":{"black":2, "cat":1},
                "duerme":{"sleeps":10, "the":1},
                "en":{"on":2, "the":1},
                "el":{"the":1},
                "sofá.":{"couch.":1}
            }
        }
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["el"]))
        self.assertEqual(2, len(self.evaluator.xword_class_pred["el"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["el"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gato"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gato"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["gato"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["negro"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["negro"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["negro"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["duerme"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["duerme"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["duerme"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["en"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["en"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["en"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sofá"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sofá"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["sofá"])
        self.assertEqual(100, self.evaluator.lang_score)

    def test_X_to_eng_using_only_alignments_exactmatch_and_inflections(self):
        path = self.create_file(
            "X_to_eng_using_only_alignments_exactmatch_and_inflection.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{
                    "src_sentence": "Ella explicó la situación con calma.",
                    "tgt_sentence_gt": "She explains the situation calmly.",
                    "tgt_sentence_pred": "She explained the situation calmly."
                }]
            })
        )
        fake_lexicon = [
            {"source_word": "gato", "target_translations": ["cat"], "src_lang": "spa", "tgt_lang": "eng"}
        ]
        fake_word_to_word_alignments = {
            0: {
                "Ella": {"She":1},
                "explicó": {"explains":1},
                "la": {"the":1},
                "situación": {"situation":1},
                "con":{"calmly.":2}, 
                "calma.":{"calmly.":1}
            }
        }
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(6, len(self.evaluator.xword_class_pred))
        self.assertEqual(6, len(self.evaluator.word_scores))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ella"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ella"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["ella"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["explicó"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["explicó"]["inflection"]))
        self.assertEqual(1, self.evaluator.word_scores["explicó"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["la"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["situación"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["situación"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["situación"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["con"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["con"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["con"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["calma"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["calma"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["calma"])
        self.assertEqual(100, self.evaluator.lang_score)

    def test_X_to_eng_using_only_alignments_exactmatch_synonym_and_incorrect(self):
        path = self.create_file(
            "X_to_eng_using_only_alignments_exactmatch_synonym_and_incorrect.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{
                    "src_sentence": "El estudiante respondió con sinceridad a todas las preguntas.",
                    "tgt_sentence_gt": "The student answered all questions sincerely.",
                    "tgt_sentence_pred": "The student answered all questions truly."
                }]
            })
        )
        fake_lexicon = [
            {
                "source_word": "gato",
                "target_translations": ["cat"],
                "src_lang": "spa",
                "tgt_lang": "eng"
            }
        ]
        fake_word_to_word_alignments = {
            0: {
                "El": {"The": 1}, 
                "estudiante": {"student": 1}, 
                "respondió": {"answered": 1}, 
                "con": {"sincerely": 1}, 
                "sinceridad": {"sincerely": 2}, 
                "a": {"to": 1}, 
                "todas": {"all": 1}, 
                "las": {"the": 1}, 
                "preguntas.": {"questions.": 1}
            }
        }
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(9, len(self.evaluator.xword_class_pred))
        self.assertEqual(9, len(self.evaluator.word_scores))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["el"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["el"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["el"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["estudiante"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["estudiante"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["estudiante"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["respondió"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["respondió"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["respondió"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["con"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["con"]["synonym"]))
        self.assertEqual(1, self.evaluator.word_scores["con"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sinceridad"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sinceridad"]["synonym"]))
        self.assertEqual(1, self.evaluator.word_scores["sinceridad"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["a"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["a"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["a"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["todas"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["todas"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["todas"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["las"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["las"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["las"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["preguntas"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["preguntas"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["preguntas"])
        self.assertAlmostEqual(88.89, self.evaluator.lang_score, places=2)

    def test_X_to_eng_using_only_alignments_gibberish(self):
        path = self.create_file(
            "X_to_eng_using_only_alignments_gibberish.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [
                    {
                        "src_sentence": "Ella cocina arroz.",
                        "tgt_sentence_gt": "",
                        "tgt_sentence_pred": "FLorny wizzle grob"
                    }
                ]
            })
        )
        fake_lexicon = [
            {"source_word": "gato", "target_translations": ["cat"], "src_lang": "spa", "tgt_lang": "eng"}
        ]
        fake_word_to_word_alignments = {
            0: {
                "Ella": {"She": 1}, 
                "cocina": {"cooks": 1}, 
                "arroz.": {"rice.": 1}
            }
        }
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(3, len(self.evaluator.xword_class_pred))
        self.assertEqual(3, len(self.evaluator.word_scores))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["arroz"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["arroz"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["arroz"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["cocina"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["cocina"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["cocina"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ella"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ella"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["ella"])
        self.assertEqual(0, self.evaluator.lang_score)

    def test_X_to_eng_using_only_alignments_multientry(self):
        path = self.create_file(
            "X_to_eng_using_only_alignments_multientry.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{
                    "src_sentence": "El gato negro duerme en el sofá.",
                    "tgt_sentence_gt": "The black cat sleeps on the couch.",
                    "tgt_sentence_pred": "The black cat sleeps on the couch."
                }, {
                    "src_sentence": "Ella explicó la situación con calma.",
                    "tgt_sentence_gt": "She explained the situation calmly.",
                    "tgt_sentence_pred": "She explains the situation calmly."
                }, {
                    "src_sentence": "El estudiante respondió con sinceridad a todas las preguntas.",
                    "tgt_sentence_gt": "The student answered all questions sincerely.",
                    "tgt_sentence_pred": "The student answered all questions truly."
                }, {
                    "src_sentence": "El gato dormía.",
                    "tgt_sentence_gt": "The cat was sleeping.",
                    "tgt_sentence_pred": "The cat was slumbering."
                }, {
                    "src_sentence": "Ella cocina arroz.",
                    "tgt_sentence_gt": "She cooks rice.",
                    "tgt_sentence_pred": "FLorny wizzle grob"
                }]
            })
        )
        fake_lexicon = [
            {"source_word": "zángano", "target_translations": ["drone"], "src_lang": "spa", "tgt_lang": "eng"}
        ]
        fake_word_to_word_alignments = {
            0: {
                "El":{"The":3, "black":2},
                "gato":{"cat":2, "couch":1},
                "negro":{"black":2, "cat":1},
                "duerme":{"sleeps":10, "the":1},
                "en":{"on":2, "the":1},
                "el":{"the":1},
                "sofá.":{"couch.":1}
            }, 1: {
                "Ella":{"She":1},
                "explicó":{"explained":1},
                "la":{"the":1},
                "situación":{"situation":1}, 
                "con":{"calmly.":2}, 
                "calma.":{"calmly.":1}
            }, 2: {
                "El": {"The": 1},
                "estudiante": {"student": 1},
                "respondió": {"answered": 1},
                "con": {"sincerely": 1},
                "sinceridad": {"sincerely": 2},
                "a": {"to": 1},
                "todas": {"all": 1},
                "las": {"the": 1},
                "preguntas.": {"questions.": 1}
            }, 3: {
                "El": {"The": 1},
                "gato": {"cat": 1},
                "dormía.": {"sleeping": 2}
            }, 4: {
                "Ella": {"She": 1},
                "cocina": {"cooks": 1},
                "arroz.": {"rice.": 1}
            }
        }
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(22, len(self.evaluator.xword_class_pred))
        self.assertEqual(22, len(self.evaluator.word_scores))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["a"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["a"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["a"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["arroz"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["arroz"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["arroz"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["calma"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["calma"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["calma"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["cocina"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["cocina"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["cocina"])
        self.assertEqual(2, len(self.evaluator.xword_class_pred["con"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["con"]["exact_match"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["con"]["synonym"]))
        self.assertEqual(1, self.evaluator.word_scores["con"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["estudiante"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["estudiante"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["estudiante"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["dormía"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["dormía"]["synonym"]))
        self.assertEqual(1, self.evaluator.word_scores["dormía"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["duerme"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["duerme"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["duerme"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["el"]))
        self.assertEqual(4, len(self.evaluator.xword_class_pred["el"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["el"])
        self.assertEqual(2, len(self.evaluator.xword_class_pred["ella"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ella"]["exact_match"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ella"]["incorrect"]))
        self.assertEqual(0.5, self.evaluator.word_scores["ella"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["en"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["en"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["en"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["explicó"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["explicó"]["inflection"]))
        self.assertEqual(1, self.evaluator.word_scores["explicó"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gato"]))
        self.assertEqual(2, len(self.evaluator.xword_class_pred["gato"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["gato"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["la"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["las"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["las"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["las"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["negro"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["negro"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["negro"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["preguntas"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["preguntas"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["preguntas"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["respondió"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["respondió"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["respondió"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sinceridad"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sinceridad"]["synonym"]))
        self.assertEqual(1, self.evaluator.word_scores["sinceridad"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["situación"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["situación"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["situación"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sofá"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sofá"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["sofá"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["todas"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["todas"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["todas"])
        self.assertAlmostEqual(84.09, self.evaluator.lang_score, places=2)

    def test_X_to_eng_using_only_alignments_multientry_no_mocking(self):
        path = self.create_file(
            "X_to_eng_using_only_alignments_multientry.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{
                    "src_sentence": "El gato negro duerme en el sofá.",
                    "tgt_sentence_gt": "The black cat sleeps on the couch.",
                    "tgt_sentence_pred": "The black cat sleeps on the couch."
                }, {
                    "src_sentence": "Ella explicó la situación con calma.",
                    "tgt_sentence_gt": "She explained the situation calmly.",
                    "tgt_sentence_pred": "She explains the situation calmly."
                }, {
                    "src_sentence": "El estudiante respondió con sinceridad a todas las preguntas.",
                    "tgt_sentence_gt": "The student answered all questions sincerely.",
                    "tgt_sentence_pred": "The student answered all questions truly."
                }, {
                    "src_sentence": "El gato dormía.",
                    "tgt_sentence_gt": "The cat was sleeping.",
                    "tgt_sentence_pred": "The cat was slumbering."
                }, {
                    "src_sentence": "Ella cocina arroz.",
                    "tgt_sentence_gt": "She cooks rice.",
                    "tgt_sentence_pred": "FLorny wizzle grob"
                }]
            })
        )
        self.evaluator.evaluate(path)
        for classification in self.evaluator.xword_class_pred.values():
            for a_list in classification.values():
                self.assertTrue(len(a_list) > 0)
        for score in self.evaluator.word_scores.values():
            self.assertTrue(score >= 0 and score <= 1)
        self.assertTrue(self.evaluator.lang_score >= 0 and self.evaluator.lang_score <= 100)

    def test_X_to_eng_priorization_of_lexicons_and_alignments_exact_match(self):
        path = self.create_file(
            "X_to_eng_prioritization_of_lexicons_and_alignments_exact_match.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{
                    "src_sentence": "El ciervo cruzó rápidamente el bosque espeso.",
                    "tgt_sentence_gt": "The deer quickly crossed the thick forest",
                    "tgt_sentence_pred": "The deer quickly crossed the thick forest."
                }]
            })
        )
        fake_lexicon = [
            {"source_word": "el", "target_translations": ["the"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "ciervo", "target_translations": ["deer"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "bosque", "target_translations": ["forest"], "src_lang": "spa", "tgt_lang": "eng"}
        ]
        fake_word_to_word_alignments = {
            0: {
                "El": {"forest.": 1},
                "cruzó": {"crossed": 1},
                "rápidamente": {"quickly": 1},
                "el": {"forest.": 1},
                "espeso.": {"thick": 1}
            }
        }
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["el"]))
        self.assertEqual(2, len(self.evaluator.xword_class_pred["el"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["el"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ciervo"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ciervo"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["ciervo"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["rápidamente"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["rápidamente"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["rápidamente"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["cruzó"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["cruzó"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["cruzó"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["bosque"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["bosque"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["bosque"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["espeso"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["espeso"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["espeso"])
        self.assertEqual(100, self.evaluator.lang_score)

    def test_X_to_eng_exactmatch_and_incorrect(self):
        path = self.create_file(
            "X_to_eng_exactmatch_and_incorrect.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{
                    "src_sentence": "El río cruza la ciudad lentamente.",
                    "tgt_sentence_gt": "The river crosses the city slowly.",
                    "tgt_sentence_pred": "The"
                }, {
                    "src_sentence": "El coche rojo estaba estacionado frente a la casa.",
                    "tgt_sentence_gt": "The red car was parked in front of the house..",
                    "tgt_sentence_pred": "Squirt."
                }]
            })
        )
        fake_lexicon = [
            {"source_word": "zángano", "target_translations": ["drone"], "src_lang": "spa", "tgt_lang": "eng"}  
        ]
        fake_word_to_word_alignments = {
            0:{
                "El":{"The":3},
            }, 1: {
                "El":{"car":1},
            }
        }
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(50, self.evaluator.lang_score)

    def test_eng_to_X_exactmatch_1(self):
        path = self.create_file(
            "eng_to_X_exactmatch_1.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "spa",
                "data": [{
                    "src_sentence": "We brought fresh bread at the store.",
                    "tgt_sentence_gt": "Compramos pan fresco en la tienda.",
                    "tgt_sentence_pred": "Compramos pan fresco en la tienda."
                }]
            })
        )
        self.evaluator.evaluate(path)
        self.assertEqual(6, len(self.evaluator.xword_class_pred))
        self.assertEqual(6, len(self.evaluator.word_scores))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["compramos"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["compramos"]["exact_match"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["pan"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["pan"]["exact_match"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["fresco"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["fresco"]["exact_match"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["en"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["en"]["exact_match"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]["exact_match"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["tienda"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["tienda"]["exact_match"]))

    def test_eng_to_X_exactmatch_2(self):
        path = self.create_file(
            "eng_to_X_exactmatch_2.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "spa",
                "data": [{
                    "src_sentence": "The cat is sleeping.",
                    "tgt_sentence_gt": "El gato está durmiendo.",
                    "tgt_sentence_pred": "El gato está durmiendo."
                }]
            })
        )
        self.evaluator.evaluate(path)
        self.assertEqual(4, len(self.evaluator.xword_class_pred))
        self.assertEqual(4, len(self.evaluator.word_scores))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["el"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["el"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["el"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gato"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gato"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["gato"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["está"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["está"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["está"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["durmiendo"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["durmiendo"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["durmiendo"])
        self.assertEqual(100, self.evaluator.lang_score)

    def test_eng_to_X_exactmatch_and_incorrect_1(self):
        path = self.create_file(
            "eng_to_X_exactmatch_and_incorrect.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "spa",
                "data": [{
                    "src_sentence": "The boy opened the window to look outside.",
                    "tgt_sentence_gt": "El niño abrió la ventana para mirar afuera.",
                    "tgt_sentence_pred": "El."
                }, {
                    "src_sentence": "The teacher wrote a letter to his students",
                    "tgt_sentence_gt": "El profesor escribió una carta a sus estudiantes.",
                    "tgt_sentence_pred": "Boop."
                }]
            })
        )
        self.evaluator.evaluate(path)
        self.assertAlmostEqual(3.33, self.evaluator.lang_score, places=2)

    def test_eng_to_X_exactmatch_and_inflection_1(self):
        path = self.create_file(
            "eng_to_X_exactmatch_and_inflection_1.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "spa",
                "data": [{
                    "src_sentence": "The streetlamp dimly lit the empty street.",
                    "tgt_sentence_gt": "La farola iluminaba débilmente la calle vacía.",
                    "tgt_sentence_pred": "El farolo ilumenaba débilmente la calle vacía."
                }]
            })
        )
        self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["farola"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["farola"]["inflection"]))
        self.assertEqual(1, self.evaluator.word_scores["farola"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["iluminaba"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["iluminaba"]["inflection"]))
        self.assertEqual(1, self.evaluator.word_scores["iluminaba"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["débilmente"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["débilmente"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["débilmente"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]))
        self.assertEqual(2, len(self.evaluator.xword_class_pred["la"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["la"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["calle"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["calle"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["calle"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["vacía"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["vacía"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["vacía"])
        self.assertEqual(100, self.evaluator.lang_score)
    
    def test_eng_to_X_exactmatch_and_inflection_2(self):
        path = self.create_file(
            "eng_to_X_exactmatch_and_inflection_2.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "spa",
                "data": [{
                    "src_sentence": "She closed the door.",
                    "tgt_sentence_gt": "Ella cerró la puerta.",
                    "tgt_sentence_pred": "Ella cerró la puerto."
                }]
            })
        )
        self.evaluator.evaluate(path)
        self.assertEqual(4, len(self.evaluator.xword_class_pred))
        self.assertEqual(4, len(self.evaluator.word_scores))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ella"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ella"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["ella"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["cerró"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["cerró"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["cerró"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["la"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["puerta"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["puerta"]["inflection"]))
        self.assertEqual(1, self.evaluator.word_scores["puerta"])
        self.assertEqual(100, self.evaluator.lang_score)

    def test_eng_to_X_gibberish(self):
        path = self.create_file(
            "eng_to_X_gibberish.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "spa",
                "data": [{
                    "src_sentence": "The cat sleeps on the chair.",
                    "tgt_sentence_gt": "El gato duerme en la silla.",
                    "tgt_sentence_pred": "Ta gortu slenfa enla chirpa."
                }]
            })
        )
        self.evaluator.evaluate(path)
        self.assertEqual(6, len(self.evaluator.xword_class_pred))
        self.assertEqual(6, len(self.evaluator.word_scores))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["el"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["el"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["el"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gato"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gato"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["gato"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["duerme"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["duerme"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["duerme"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["en"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["en"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["en"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["la"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["silla"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["silla"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["silla"])
        self.assertEqual(0, self.evaluator.lang_score)

    def test_eng_to_X_multientry(self):
        path = self.create_file(
            "eng_to_X_multientry.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "spa",
                "data": [
                    {
                        "src_sentence": "The cat is sleeping.",
                        "tgt_sentence_gt": "El gato está durmiendo.",
                        "tgt_sentence_pred": "El gato está durmiendo."
                    }, {
                        "src_sentence": "She closed the door.",
                        "tgt_sentence_gt": "Ella cerró la puerta.",
                        "tgt_sentence_pred": "Ella cerró la puerto."
                    }, {
                        "src_sentence": "The cat sleeps on the chair.",
                        "tgt_sentence_gt": "El gato duerme en la silla.",
                        "tgt_sentence_pred": "Ta gortu slenfa enla chirpa."
                    }
                ]
            })
        )
        self.evaluator.evaluate(path)
        self.assertEqual(11, len(self.evaluator.xword_class_pred))
        self.assertEqual(11, len(self.evaluator.word_scores))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["cerró"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["cerró"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["cerró"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["durmiendo"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["durmiendo"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["durmiendo"])
        self.assertEqual(2, len(self.evaluator.xword_class_pred["el"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["el"]["exact_match"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["el"]["incorrect"]))
        self.assertEqual(0.5, self.evaluator.word_scores["el"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["duerme"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["duerme"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["duerme"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ella"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ella"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["ella"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["en"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["en"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["en"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["está"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["está"]["exact_match"]))
        self.assertEqual(1, self.evaluator.word_scores["está"])
        self.assertEqual(2, len(self.evaluator.xword_class_pred["gato"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gato"]["exact_match"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gato"]["incorrect"]))
        self.assertEqual(0.5, self.evaluator.word_scores["gato"])
        self.assertEqual(2, len(self.evaluator.xword_class_pred["la"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]["exact_match"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["la"]["incorrect"]))
        self.assertEqual(0.5, self.evaluator.word_scores["la"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["puerta"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["puerta"]["inflection"]))
        self.assertEqual(1, self.evaluator.word_scores["puerta"])
        self.assertEqual(1, len(self.evaluator.xword_class_pred["silla"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["silla"]["incorrect"]))
        self.assertEqual(0, self.evaluator.word_scores["silla"])
        self.assertAlmostEqual(59.09, self.evaluator.lang_score, places=2)
