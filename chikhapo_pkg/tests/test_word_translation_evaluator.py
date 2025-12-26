import json
import unittest
from unittest.mock import patch
from chikhapo import Evaluator
from .base_evaluator_test import BaseEvaluatorTest

class TestWordTranslationEvaluator(BaseEvaluatorTest):
    """Unit tests for the WordTranslationEvaluator class."""

    def setUp(self):
        super().setUp()
        self.evaluator = Evaluator("word_translation")

    def test_no_prediction(self):
        path = self.create_file(
            "missing_prediction_in_entry.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "a", "prediction": "a"},
                         {"word": "b"}],
            }),
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn("prediction was not specified in", str(ctx.exception))

    def test_exact_match_spa_eng_1(self):
        """Should correctly classify an exact match between spa and eng."""
        path = self.create_file(
            "exact_match_spa_eng_1.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "gatos", "prediction": "cat."}],
            }),
        )
        fake_lexicon = [
            {"source_word": "escribieron", "target_translations": ["write"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "feliz", "target_translations": ["happy"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "gatos", "target_translations": ["cat"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "perro", "target_translations": ["dog"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "libro", "target_translations": ["book"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "ventana", "target_translations": ["window"], "src_lang": "spa", "tgt_lang": "eng"},
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(["cat"], self.evaluator.xword_class_pred["gatos"]["exact_match"])
        self.assertEqual(1, self.evaluator.word_scores["gatos"])
        self.assertEqual(100, self.evaluator.lang_score)

    def test_exact_match_spa_eng_2(self):
        path = self.create_file(
            "exact_match_spa_eng_2.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "el", "prediction": "the."}],
            }),
        )
        fake_lexicon = [
            {"source_word": "escriberon", "target_translations": ["write"]},
            {"source_word": "feliz", "target_translations": ["happy"]},
            {"source_word": "gatos", "target_translations": ["cat"]},
            {"source_word": "perro", "target_translations": ["dog"]},
            {"source_word": "libro", "target_translations": ["book"]},
            {"source_word": "ventana", "target_translations": ["window"]},
            {"source_word": "el", "target_translations": ["the", "write", "happy", "cat", "dog"]},
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, self.evaluator.word_scores["el"])
        self.assertEqual(100, self.evaluator.lang_score)

    def test_inflection_eng_spa(self):
        """Should correctly identify inflectional relationships."""
        path = self.create_file(
            "inflection_eng_spa.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "spa",
                "data": [{"word": "Egyptian", "prediction": "Egipto."}],
            }),
        )
        fake_lexicon = [
            {
                "source_word": "egyptian",
                "target_translations": [
                    "egipcio","de egipto","egipcia","egipciaco","egipcíaco","egipciano","lengua egipcia",
                ],
                "src_lang": "eng",
                "tgt_lang": "spa",
            }
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(["egipto"], self.evaluator.xword_class_pred["egipcia"]["inflection"])
        self.assertEqual(1, self.evaluator.word_scores["egipcia"])
        self.assertEqual(100, self.evaluator.lang_score)

    def test_inflection_spa_eng(self):
        path = self.create_file(
            "inflection_spa_eng.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "escribieron", "prediction": "wrote."}]
            })
        )
        fake_lexicon = [
            {"source_word": "escribieron", "target_translations": ["write"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "feliz", "target_translations": ["happy"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "gatos", "target_translations": ["cat"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "perro", "target_translations": ["dog"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "libro", "target_translations": ["book"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "ventana", "target_translations": ["window"], "src_lang": "spa", "tgt_lang": "eng"},
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["escribieron"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["escribieron"]["inflection"]))
        self.assertEqual(1, self.evaluator.word_scores["escribieron"])
        self.assertEqual(100, self.evaluator.lang_score)

    def test_substring_spa_eng(self):
        path = self.create_file(
            "substring_spa_eng.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "gatos", "prediction": "the answer is cat."}]
            })
        )
        fake_lexicon = [
            {"source_word": "escribieron", "target_translations": ["write"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "feliz", "target_translations": ["happy"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "gatos", "target_translations": ["cat"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "perro", "target_translations": ["dog"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "libro", "target_translations": ["book"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "ventana", "target_translations": ["window"], "src_lang": "spa", "tgt_lang": "eng"},
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gatos"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gatos"]["substring"]))
        self.assertEqual(1, len(self.evaluator.word_scores))
        self.assertEqual(1, self.evaluator.word_scores["gatos"])
        self.assertEqual(100, self.evaluator.lang_score)

    def test_inflection_within_substring(self):
        path = self.create_file(
            "inflection_within_substring_spa_eng.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "fondo", "prediction": "the answer is backgriund."}]
            })
        )
        fake_lexicon = [
            {"source_word": "fondo", "target_translations": ["background", "bottom", "fund", "depth"]}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["fondo"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["fondo"]["inflection_within_substring"]))
        self.assertEqual(1, len(self.evaluator.word_scores))
        self.assertEqual(1, self.evaluator.word_scores["fondo"])
        self.assertEqual(100, self.evaluator.lang_score)

    def test_synonym_spa_eng(self):
        path = self.create_file(
            "synonym_spa_eng.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "perro", "prediction": "hound."}]
            })
        )
        fake_lexicon = [
            {"source_word": "escribieron", "target_translations": ["write"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "feliz", "target_translations": ["happy"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "gatos", "target_translations": ["cat"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "perro", "target_translations": ["dog"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "libro", "target_translations": ["book"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "ventana", "target_translations": ["window"], "src_lang": "spa", "tgt_lang": "eng"},
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["perro"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["perro"]["synonym"]))
        self.assertEqual(100, self.evaluator.lang_score)

    def test_echo_spa_eng(self):
        path = self.create_file(
            "echo_spa_eng.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "gatos", "prediction": "gatos."}]
            })
        )
        fake_lexicon = [
            {"source_word": "escriberon", "target_translations": ["write"]},
            {"source_word": "feliz", "target_translations": ["happy"]},
            {"source_word": "gatos", "target_translations": ["cat"]},
            {"source_word": "perro", "target_translations": ["dog"]},
            {"source_word": "libro", "target_translations": ["book"]},
            {"source_word": "ventana", "target_translations": ["window"]},
            {"source_word": "el", "target_translations": ["the", "write", "happy", "cat", "dog"]},
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gatos"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gatos"]["echo"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_outputted_in_source_language_spa_end(self):
        path = self.create_file(
            "outputted_in_source_language_spa_eng.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "gatos", "prediction": "perro."}]
            })
        )
        fake_lexicon = [
            {"source_word": "escriberon", "target_translations": ["write"]},
            {"source_word": "feliz", "target_translations": ["happy"]},
            {"source_word": "gatos", "target_translations": ["cat"]},
            {"source_word": "perro", "target_translations": ["dog"]},
            {"source_word": "libro", "target_translations": ["book"]},
            {"source_word": "ventana", "target_translations": ["window"]},
            {"source_word": "el", "target_translations": ["the", "write", "happy", "cat", "dog"]},
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gatos"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gatos"]["outputted_in_source_language"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_zul_eng(self):
        path = self.create_file(
            "gibberish_zul_eng.json",
            json.dumps({
                "src_lang": "zul",
                "tgt_lang": "eng",
                "data": [{"word": "isigabavu", "prediction": "I have no idea what this word means. It doesn't seem to be a word in any of the languages I know."}]
            })
        )
        fake_lexicon=[
            {"source_word": "isigabavu", "target_translations": ["effort", "endeavour"]}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["isigabavu"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["isigabavu"]["gibberish"]))
        self.assertEqual(1, len(self.evaluator.word_scores))
        self.assertEqual(0, self.evaluator.word_scores["isigabavu"])
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_spa_eng(self):
        path = self.create_file(
            "gibberish_spa_eng.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "perro", "prediction": "qwerty."}]
            })
        )
        fake_lexicon = [
            {"source_word": "escribieron", "target_translations": ["write"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "feliz", "target_translations": ["happy"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "gatos", "target_translations": ["cat"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "perro", "target_translations": ["dog"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "libro", "target_translations": ["book"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "ventana", "target_translations": ["window"], "src_lang": "spa", "tgt_lang": "eng"},
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["perro"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["perro"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_equivalence_spa_eng(self):
        """Should correctly evaluate partial equivalence between spa and eng."""
        path = self.create_file(
            "equivalence.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [
                    {"word": "feliz", "prediction": "qwerty."},
                    {"word": "perro", "prediction": "hound."},
                    {"word": "escribieron", "prediction": "wrote."},
                    {"word": "libro", "prediction": "libro."},
                    {"word": "gatos", "prediction": "a cat."},
                    {"word": "ventana", "prediction": "window."},
                ],
            }),
        )
        fake_lexicon = [
            {"source_word": "escribieron", "target_translations": ["write"]},
            {"source_word": "feliz", "target_translations": ["happy"]},
            {"source_word": "gatos", "target_translations": ["cat"]},
            {"source_word": "perro", "target_translations": ["dog"]},
            {"source_word": "libro", "target_translations": ["book"]},
            {"source_word": "ventana", "target_translations": ["window"]},
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(6, len(self.evaluator.xword_class_pred))
        self.assertAlmostEqual(66.66667, self.evaluator.lang_score, places=3)
    
    def test_multiple_src_words_correct_and_incorrect(self):
        path = self.create_file(
            "multiple_src_words_correct_and_incorrect.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "spa",
                "data": [
                    {"word": "story", "prediction": "echo."},
                    {"word": "history", "prediction": "historia."},
                ]
            })
        )
        fake_lexicon = [
            {"source_word": "story", "target_translations": ["historia", "cuento", "piso", "planta"]},
            {"source_word": "history", "target_translations": ["historia"]}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(0.5, self.evaluator.word_scores["historia"])
        self.assertEqual(0, self.evaluator.word_scores["cuento"])
        self.assertEqual(0, self.evaluator.word_scores["piso"])
        self.assertEqual(0, self.evaluator.word_scores["planta"])
        self.assertEqual(12.5, self.evaluator.lang_score)
