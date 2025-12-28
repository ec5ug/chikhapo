import json
from chikhapo import Evaluator
from .base_evaluator_test import BaseEvaluatorTest

class TestEvaluator(BaseEvaluatorTest):
    def setUp(self):
        super().setUp()
        self.evaluator = Evaluator("word_translation")
    
    def test_improper_file_extension(self):
        """Should raise an Exception when file is not a JSON file."""
        path = self.create_file("wrong_file_extension.txt", "lorem ipsum")
        with self.assertRaises(Exception) as ctx:
            self.evaluator.read_prediction_file(path)
        self.assertIn("not a JSON file", str(ctx.exception))

    def test_missing_source_language(self):
        """Should raise an Exception when src_lang is missing."""
        path = self.create_file(
            "missing_source_language.json",
            json.dumps({
                "tgt_lang": "eng",
                "data": [{"word": "a", "prediction": "a"}],
            }),
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.read_prediction_file(path)
        self.assertIn('The key "src_lang" is not specified.', str(ctx.exception))

    def test_improper_source_language(self):
        path = self.create_file(
            "improper_source_language.json",
            json.dumps({
                "src_lang": 1,
                "tgt_lang": "eng",
                "data": [{"word": "a", "prediction": "a"}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.read_prediction_file(path)
        self.assertIn("The source language should be specified as a string.", str(ctx.exception))

    def test_missing_target_language(self):
        path = self.create_file(
            "missing_target_language.json",
            json.dumps({
                "src_lang": "spa",
                "data": [{"word": "a", "prediction": "b"}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.read_prediction_file(path)
        self.assertIn("The key \"tgt_lang\" is not specified. Please specify the key to the target language.", str(ctx.exception))

    def test_improper_target_language(self):
        path = self.create_file(
            "improper_target_language.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": 2,
                "data": [{"word": "a", "prediction": "a"}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.read_prediction_file(path)
        self.assertIn("The target language should be specified as a string.", str(ctx.exception))

    def test_src_lang_all(self):
        path = self.create_file(
            "improper_target_language.json",
            json.dumps({
                "src_lang": "all",
                "tgt_lang": "eng",
                "data": [{"word": "a", "prediction": "a"}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.read_prediction_file(path)
        self.assertIn("This function can only evaluate data from one translation. You will have to split your data by language pair and evaluate each split separately.", str(ctx.exception))

    def test_tgt_lang_all(self):
        path = self.create_file(
            "improper_target_language.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "all",
                "data": [{"word": "a", "prediction": "a"}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.read_prediction_file(path)
        self.assertIn("This function can only evaluate data from one translation. You will have to split your data by language pair and evaluate each split separately.", str(ctx.exception))

    def test_no_data(self):
        path = self.create_file(
            "no_data.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.read_prediction_file(path)
        self.assertIn("The key \"data\" is not specified. Please specify the key to data.", str(ctx.exception))

    def test_improper_data(self):
        path = self.create_file(
            "improper_data.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": {"word": "a", "prediction": "a"}
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.read_prediction_file(path)
        self.assertIn("The data you provided does not exist as a list. Please specify the data as a list", str(ctx.exception))

    def test_equivalence(self):
        path = self.create_file(
            "equivalence.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [
                    {"word": "mesa", "prediction": "cloud"},
                    {"word": "caminar", "prediction": "to swim"},
                    {"word": "ventana", "prediction": "book"}
                ]
            })
        )
        self.evaluator.read_prediction_file(path)
        self.assertEqual("spa", self.evaluator.src_lang)
        self.assertEqual("eng", self.evaluator.tgt_lang)
        self.assertEqual([
            {"word": "mesa", "prediction": "cloud"},
            {"word": "caminar", "prediction": "to swim"},
            {"word": "ventana", "prediction": "book"}
        ], self.evaluator.data)
        