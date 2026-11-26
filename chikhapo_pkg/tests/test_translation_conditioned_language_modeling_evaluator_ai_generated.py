import json
import os
import pprint
import tempfile
import unittest
from unittest.mock import patch

from .base_evaluator_test import BaseEvaluatorTest
from chikhapo import Evaluator

real_data_path = "/share/data/speech-lang/users/echang/chikhapo/chikhapo_pkg/tests/raw_test_data/translation_conditioned_language_modeling/appendix/parsed_responses/"
x_to_eng_tolerance = 6 # due to changes in how fast_align is trained
eng_to_x_tolerance = 2

class TestTranslationConditionedLanguageModelingEvaluator(BaseEvaluatorTest):
    def setUp(self):
        super().setUp()
        self.evaluator = Evaluator("translation_conditioned_language_modeling")
    
    def test_no_src_sentence(self):
        path = self.create_file(
            "missing_src_sentence.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"tgt_sentence_gt": "a", "next_word_to_predict": "a", "probability": 0}]
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
                "data": [{"src_sentence": 1, "tgt_sentence_gt": "a", "next_word_to_predict": "a", "probability": 0}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn(": The source sentence in \"src_sentence\" should be stored as a string, not ", str(ctx.exception))

    def test_no_tgt_sentence_gt(self):
        path = self.create_file(
            "no_tgt_sentence_gt.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": "a", "next_word_to_predict": "a", "probability": 0}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn("The key \"tgt_sentence_gt\" is not found in ", str(ctx.exception))

    def test_improper_tgt_sentence_gt(self):
        path = self.create_file(
            "improper_tgt_sentence_gt.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": "a", "tgt_sentence_gt": 1, "next_word_to_predict": "a", "probability": 0}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn(": The ground-truth target sentence in \"tgt_sentence_gt\" should be stored as a string, not ", str(ctx.exception))
    
    def test_no_next_word_to_predict(self):
        path = self.create_file(
            "no_next_word_to_predict.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": "a", "tgt_sentence_gt": "b", "probability": 0}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn("The key \"next_word_to_predict\" is not found in ", str(ctx.exception))
    
    def test_improper_next_word_to_predict(self):
        path = self.create_file(
            "improper_next_word_to_predict.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": "b", "tgt_sentence_gt": "a", "next_word_to_predict": 1, "probability": 0}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn(": The next word to predict in \"next_word_to_predict\" should be stored as a string, not ", str(ctx.exception))
    
    def test_no_probability(self):
        path = self.create_file(
            "no_probability.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": "a b", "tgt_sentence_gt": "c d", "next_word_to_predict": "c"}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn("The key \"probability\" is not found in ", str(ctx.exception))

    def test_improper_probability_not_float(self):
        path = self.create_file(
            "no_probability.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": "a b", "tgt_sentence_gt": "c d", "next_word_to_predict": "c", "probability": "1"}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn(": The value of the key \"probability\" should be represented as a float.", str(ctx.exception))
    
    def test_improper_probability_less_than_0(self):
        path = self.create_file(
            "no_probability.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": "a b", "tgt_sentence_gt": "c d", "next_word_to_predict": "c", "probability": -0.05}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn("The probability measure ", str(ctx.exception))
        self.assertIn(" should be in the interval of 0 and 1.", str(ctx.exception))

    def test_improper_probability_more_than_1(self):
        path = self.create_file(
            "no_probability.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"src_sentence": "a b", "tgt_sentence_gt": "c d", "next_word_to_predict": "c", "probability": 1.05}]
            })
        )
        with self.assertRaises(Exception) as ctx:
            self.evaluator.evaluate(path)
        self.assertIn("The probability measure ", str(ctx.exception))
        self.assertIn(" should be in the interval of 0 and 1.", str(ctx.exception))

    def test_english_word_has_translation_in_lexicon_but_translations_are_not_found_in_source_sentence(self):
        path = self.create_file(
            "english_word_has_translation_in_lexicon_but_translations_are_not_found_in_source_sentence.json",
            json.dumps({
                "src_lang": "fra",
                "tgt_lang": "eng",
                "data": [
                    {
                        "src_sentence": "Ils vivent dans une demeure ancienne.",
                        "tgt_sentence_gt": "They live in an old house.",
                        "next_word_to_predict": "house ",
                        "probability": 0.2
                    }
                ]
            })
        )
        fake_lexicon = [
            {"source_word": "house", "target_translations": ["maison", "logement", "habitation", "résidence", "pavilion", "bâtisse"], "src_lang": "fra", "tgt_lang": "eng"}
        ]
        fake_word_to_word_alignments = {
            0: {
                "They": {"Ils": 1},
                "live": {"vivent": 1},
                "in": {"dans": 1},
                "an": {"une": 1},
                "old": {"ancienne.": 1}
            }
        }
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(0, len(self.evaluator.word_scores))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_english_word_has_no_translation_in_lexicon_nor_alignments(self):
        path = self.create_file(
            "english_word_has_no_translation_in_lexicon.json",
            json.dumps({
                "src_lang": "fra",
                "tgt_lang": "eng",
                "data": [{
                    "src_sentence": "Les enfants jouent dans le jardin.",
                    "tgt_sentence_gt": "The children are playing in the garden.",
                    "next_word_to_predict": "children ",
                    "probability": 0.5
                }]
            })
        )
        fake_lexicon = [
            {"source_word": "light", "target_translations": ["lumière", "léger", "légère", "clair", "claire", "allégé", "allégée", "faible"], "src_lang": "fra", "tgt_lang": "eng"}
        ]
        fake_word_to_word_alignments = {
            0: {
                "The": {"Les": 1},
                "playing": {"jouent": 1},
                "in": {"dans": 1},
                "the": {"le": 1},
                "garden.": {"jardin.": 1}
            }
        }
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(0, len(self.evaluator.word_scores))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_lexicon_has_multiple_translations_found_in_sentence(self):
        path = self.create_file(
            "lexicon_has_multiple_translations_found_in_sentence.json",
            json.dumps({
                "src_lang": "fra",
                "tgt_lang": "eng",
                "data": [
                    {
                        "src_sentence": "La maison et la résidence sont situées près de la rivière.",
                        "tgt_sentence_gt": "The house and the residence are located near the river.", 
                        "next_word_to_predict": "residence ", 
                        "probability": 0}
                ]
            })
        )
        fake_lexicon = [
            {
                "source_word": "residence",
                "target_translations": ["maison", "logement", "habitation", "résidence", "pavillon", "bâtisse"],
                "src_lang": "fra",
                "tgt_lang": "eng"}
        ]
        fake_word_to_word_alignments = {
            0: {
                "The": {"La": 1},
                "house": {"maison": 1},
                "and": {"et": 1},
                "the": {"la": 1},
            }
        }
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(2, len(self.evaluator.word_scores))
        self.assertEqual(0, self.evaluator.word_scores["résidence"])
        self.assertEqual(0, self.evaluator.word_scores["maison"])
        self.assertEqual(0, self.evaluator.lang_score)

    def test_lexicon_has_improperly_cased_and_spaced_translations(self):
        path = self.create_file(
            "lexicon_has_improperly_cased_and_space_translations.json",
            json.dumps({
                "src_lang": "fra",
                "tgt_lang": "eng",
                "data": [
                    {"src_sentence": "La maison et la résidence sont situées près de la rivière.", "tgt_sentence_gt": "The house and the residence are located near the river.", "next_word_to_predict": "residence", "probability": 0}
                ]
            })
        )
        fake_lexicon = [{"source_word": "residence", "target_translations": ["   MaiSon  ", "LOGEMENT", "Habitation", "   réSIdence ", "PavillON", "bÂtIsSe"], "src_lang": "fra", "tgt_lang": "eng"}]
        fake_word_to_word_alignments = {
            0: {
                "The": {"La": 1},
                "house": {"maison": 1},
                "and": {"et": 1},
                "the": {"la": 1},
            }
        }
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(2, len(self.evaluator.word_scores))
        self.assertEqual(0, self.evaluator.word_scores["résidence"])
        self.assertEqual(0, self.evaluator.word_scores["maison"])
        self.assertEqual(0, self.evaluator.lang_score)

    def test_lexicon_has_multiword_translations_found_in_sentence(self):
        path =self.create_file(
            "lexicons_has_multiword_translations_found_in_sentence.json",
            json.dumps({
                "src_lang": "fra",
                "tgt_lang": "eng",
                "data": [{
                    "src_sentence": "Il a quitté la maison tôt ce matin.",
                    "tgt_sentence_gt": "He left the house early this morning.",
                    "next_word_to_predict": "house ",
                    "probability": 0.2
                }]
            })
        )
        fake_lexicon = [{"source_word": "house", "target_translations": ["la maison", "le logement", "l'habitation"], "src_lang": "fra", "tgt_lang": "eng"}]
        fake_word_to_word_alignments = {
            0: {
                "He": {"Il": 1},
                "left": {"quitte": 1},
                "the": {"la": 1},
            }
        }
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.word_scores))
        self.assertEqual(0.2, self.evaluator.word_scores["la maison"])
        self.assertEqual(20, self.evaluator.lang_score)

    def test_lexicon_has_inflected_translation_found_in_src_sentence(self):
        path =self.create_file(
            "lexicon_has_inflected_translation_found_in_src_sentence.json",
            json.dumps({
                "src_lang": "fra",
                "tgt_lang": "eng",
                "data": [{
                    "src_sentence": "Il a quitté la maison tôt ce matin.",
                    "tgt_sentence_gt": "He left the house early this morning.",
                    "next_word_to_predict": "left ",
                    "probability": 0.2
                }]
            })
        )
        fake_lexicon = [{"source_word": "left", "target_translations": ["quittér", "partir", "s'en aller"], "src_lang": "fra", "tgt_lang": "eng"}]
        fake_word_to_word_alignments = {
            0: {
                "He": {"Il": 1}
            }
        }
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.word_scores))
        self.assertEqual(0.2, self.evaluator.word_scores["quittér"])
        self.assertEqual(20, self.evaluator.lang_score)

    def test_lexicon_has_inflected_multiword_translation_found_in_src_sentence(self):
        path =self.create_file(
            "lexicon_has_inflected_multiword_translation_found_in_src_sentence.json",
            json.dumps({
                "src_lang": "fra",
                "tgt_lang": "eng",
                "data": [{
                    "src_sentence": "Il a quitté la maison tôt ce matin.",
                    "tgt_sentence_gt": "He left the house early this morning.",
                    "next_word_to_predict": "left ",
                    "probability": 0.2
                }]
            })
        )
        fake_lexicon = [{"source_word": "left", "target_translations": ["requittèrent", "s’étaient quittés", "abandonné"], "src_lang": "fra", "tgt_lang": "eng"}]
        fake_word_to_word_alignments = {
            0: {
                "He": {"Il": 1}
            }
        }
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.word_scores))
        self.assertEqual(0.2, self.evaluator.word_scores["s’étaient quittés"])
        self.assertEqual(20, self.evaluator.lang_score)

    def test_multientry_X_to_eng_uses_trans_dict(self):
        path =self.create_file(
            "multientry_X_to_eng_uses_trans_dict.json",
            json.dumps({
                "src_lang": "fra",
                "tgt_lang": "eng",
                "data": [
                    {
                        "src_sentence": "La maison et la résidence sont situées près de la rivière.",
                        "tgt_sentence_gt": "The house and the residence are located near the river.",
                        "next_word_to_predict": "The ",
                        "probability": 0.1
                    }, {
                        "src_sentence": "La maison et la résidence sont situées près de la rivière.",
                        "tgt_sentence_gt": "The house and the residence are located near the river.",
                        "next_word_to_predict": "house ",
                        "probability": 0.2
                    }, {
                        "src_sentence": "La maison et la résidence sont situées près de la rivière.",
                        "tgt_sentence_gt": "The house and the residence are located near the river.",
                        "next_word_to_predict": "and ",
                        "probability": 0.3
                    }, {
                        "src_sentence": "La maison et la résidence sont situées près de la rivière.",
                        "tgt_sentence_gt": "The house and the residence are located near the river.",
                        "next_word_to_predict": "the ",
                        "probability": 0.4
                    }, {
                        "src_sentence": "La maison et la résidence sont situées près de la rivière.",
                        "tgt_sentence_gt": "The house and the residence are located near the river.",
                        "next_word_to_predict": "residence ",
                        "probability": 0.2
                    },
                ]
            })
        )
        fake_lexicon = [
            {"source_word": "the", "target_translations": ["le", "la", "les", "l'"], "src_lang": "fra", "tgt_lang": "eng"},
            {"source_word": "house", "target_translations": ["maison", "foyer", "demeure", "habitation", "logement"], "src_lang": "fra", "tgt_lang": "eng"},
            {"source_word": "and", "target_translations": ["et", "ainsi que", "de plus", "puis", "voire"], "src_lang": "fra", "tgt_lang": "eng"},
            {"source_word": "residence", "target_translations": ["maison", "logement", "habitation", "résidence", "pavillon", "bâtisse"], "src_lang": "fra", "tgt_lang": "eng"}
        ]
        fake_word_to_word_alignments = {
            0: {
                "are": {"sont": 1}
            }, 1: {
                "near": {"près": 1}
            }, 2: {
                "of": {"de": 1}
            }, 3: {
                "river": {"rivière": 1}
            }, 4: {
                "located": {"situées": 1}
            }
        }
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon), patch.object(self.evaluator, "get_statistical_alignments", return_value=fake_word_to_word_alignments):
            self.evaluator.evaluate(path)
        self.assertEqual(5, len(self.evaluator.word_scores))
        self.assertEqual(0.25, self.evaluator.word_scores["la"])
        self.assertEqual(0.2, self.evaluator.word_scores["maison"])
        self.assertEqual(0.3, self.evaluator.word_scores["et"])
        self.assertEqual(0.2, self.evaluator.word_scores["résidence"])
        self.assertEqual(0.3, self.evaluator.word_scores["de plus"])
    
    def test_multientry_X_to_eng_uses_trans_dict_no_mocking(self):
        path =self.create_file(
            "multientry_X_to_eng_uses_trans_dict_no_mocking.json",
            json.dumps({
                "src_lang": "fra",
                "tgt_lang": "eng",
                "data": [
                    {
                        "src_sentence": "La maison et la résidence sont situées près de la rivière.",
                        "tgt_sentence_gt": "The house and the residence are located near the river.",
                        "next_word_to_predict": "The ",
                        "probability": 0.1
                    }, {
                        "src_sentence": "La maison et la résidence sont situées près de la rivière.",
                        "tgt_sentence_gt": "The house and the residence are located near the river.",
                        "next_word_to_predict": "house ",
                        "probability": 0.2
                    }, {
                        "src_sentence": "La maison et la résidence sont situées près de la rivière.",
                        "tgt_sentence_gt": "The house and the residence are located near the river.",
                        "next_word_to_predict": "and ",
                        "probability": 0.3
                    }, {
                        "src_sentence": "La maison et la résidence sont situées près de la rivière.",
                        "tgt_sentence_gt": "The house and the residence are located near the river.",
                        "next_word_to_predict": "the ",
                        "probability": 0.4
                    }, {
                        "src_sentence": "La maison et la résidence sont situées près de la rivière.",
                        "tgt_sentence_gt": "The house and the residence are located near the river.",
                        "next_word_to_predict": "residence ",
                        "probability": 0.2
                    },
                ]
            })
        )
        self.evaluator.evaluate(path)
        for src_word in self.evaluator.word_scores:
            self.assertTrue(self.evaluator.word_scores[src_word] > 0)
        self.assertTrue(self.evaluator.lang_score >= 0 and self.evaluator.lang_score <= 100)

    def test_singleentry_eng_to_X_1(self):
        path = self.create_file(
            "singleentry_eng_to_X.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "fra",
                "data": [{
                    "src_sentence": "The cat sleeps on the sofa.",
                    "tgt_sentence_gt": "Le chat dort sur le canapé.",
                    "next_word_to_predict": "canapé",
                    "probability": 0
                }]
            })
        )
        self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.word_scores))
        self.assertEqual(0, self.evaluator.word_scores["canapé"])
        self.assertEqual(0, self.evaluator.lang_score)

    def test_singleentry_eng_to_X_2(self):
        path = self.create_file(
            "singleentry_eng_to_X_2.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "fra",
                "data": [
                    {
                        "src_sentence": "The girl opened the door slowly.",
                        "tgt_sentence_gt": "La fille ouvrit la porte lentement.",
                        "next_word_to_predict": "ouvert",
                        "probability": 0.95
                    }
                ]
            })
        )
        self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.word_scores))
        self.assertEqual(0.95, self.evaluator.word_scores["ouvert"])
        self.assertEqual(95, self.evaluator.lang_score)

    def test_singleword_eng_to_X_1(self):
        path = self.create_file(
            "multientry_eng_to_X_1.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "fra",
                "data": [
                    {
                        "src_sentence": "The cat is sleeping.",
                        "tgt_sentence_gt": "Le chat dort sur le canapé.",
                        "next_word_to_predict": "chat",
                        "probability": 0.75
                    },                     {
                        "src_sentence": "I saw a cat in the garden.",
                        "tgt_sentence_gt": "J'ai vu un chat dans le jardin.",
                        "next_word_to_predict": "chat",
                        "probability": 0.25
                    }
                ]
            })
        )
        self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.word_scores))
        self.assertEqual(0.5, self.evaluator.word_scores["chat"])

    def test_singleword_eng_to_X_2(self):
        path = self.create_file(
            "singleword_eng_to_X_2.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "fra",
                "data": [
                    {
                        "src_sentence": "The boy gave the book to the boy.",
                        "tgt_sentence_gt": "Le garçon a donné le livre au garçon.",
                        "next_word_to_predict": "garçon",
                        "probability": 0.1
                    }, {
                        "src_sentence": "The boy gave the book to the boy.",
                        "tgt_sentence_gt": "Le garçon a donné le livre au garçon.",
                        "next_word_to_predict": "garçon",
                        "probability": 0.8
                    }
                ]
            })
        )
        self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.word_scores))
        self.assertEqual(0.45, self.evaluator.word_scores["garçon"])

    def test_multientry_eng_to_X(self):
        path = self.create_file(
            "multientry_eng_to_X.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "fra",
                "data": [
                    {
                        "src_sentence": "The cat sleeps on the sofa.",
                        "tgt_sentence_gt": "Le chat dort sur le canapé.",
                        "next_word_to_predict": "canapé",
                        "probability": 0
                    }, {
                        "src_sentence": "The cat is sleeping.",
                        "tgt_sentence_gt": "Le chat dort.",
                        "next_word_to_predict": "chat",
                        "probability": 0.75
                    }, {
                        "src_sentence": "I saw a cat in the garden.",
                        "tgt_sentence_gt": "J'ai vu un chat dans le jardin.",
                        "next_word_to_predict": "chat",
                        "probability": 0.25
                    }, {
                        "src_sentence": "The boy gave the book to the boy.",
                        "tgt_sentence_gt": "Le garçon a donné le livre au garçon.",
                        "next_word_to_predict": "garçon",
                        "probability": 0.1
                    }, {
                        "src_sentence": "The boy gave the book to the boy.",
                        "tgt_sentence_gt": "Le garçon a donné le livre au garçon.",
                        "next_word_to_predict": "garçon",
                        "probability": 0.8
                    }
                ]
            })
        )
        self.evaluator.evaluate(path)
        self.assertEqual(3, len(self.evaluator.word_scores))
        self.assertEqual(0, self.evaluator.word_scores["canapé"])
        self.assertEqual(0.5, self.evaluator.word_scores["chat"])
        self.assertEqual(0.45, self.evaluator.word_scores["garçon"])
        self.assertAlmostEqual(31.67, self.evaluator.lang_score, places=2)
