import json
from unittest.mock import patch
from chikhapo import Evaluator
from .base_evaluator_test import BaseEvaluatorTest

class TestWordTranslationWithContextEvaluator(BaseEvaluatorTest):
    """Unit tests for the WordTranslationWithContextEvaluator class"""

    def setUp(self):
        super().setUp()
        self.evaluator = Evaluator("word_translation_with_context")

    def test_exact_match_spa_eng_1(self):
        path = self.create_file(
            "exact_match_spa_eng_1.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "gatos", "prediction": "cat"}],
            }),
        )
        fake_lexicon = [
            {"source_word": "gatos", "target_translations": ["cat"], "src_lang": "spa", "tgt_lang": "eng"},
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gatos"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gatos"]["exact_match"]))
        self.assertEqual(100, self.evaluator.lang_score)
    
    def test_exact_match_spa_eng_2(self):
        path = self.create_file(
            "exact_match_spa_eng_2.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "ancha", "prediction": "much"}]
            })
        )
        fake_lexicon = [
            {"source_word": "ancha", "target_translations": ["much"], "src_lang": "spa", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ancha"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ancha"]["exact_match"]))
        self.assertEqual(100, self.evaluator.lang_score)

    def test_exact_match_sun_eng_1(self):
        path = self.create_file(
            "exact_match_sun_eng_1.json",
            json.dumps({
                "src_lang": "sun",
                "tgt_lang": "eng",
                "data": [{"word": "iyeu", "prediction": "they"}]
            })
        )
        fake_lexicon = [
            {"source_word": "iyeu", "target_translations": ["they"], "src_lang": "sun", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["iyeu"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["iyeu"]["exact_match"]))
        self.assertEqual(100, self.evaluator.lang_score)
    
    def test_exact_match_sun_eng_2(self):
        path = self.create_file(
            "exact_match_sun_eng_2.json",
            json.dumps({
                "src_lang": "sun",
                "tgt_lang": "eng",
                "data": [{"word": "lain", "prediction": "other"}]
            })
        )
        fake_lexicon = [
            {"source_word": "lain", "target_translations": ["other"], "src_lang": "sun", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["lain"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["lain"]["exact_match"]))
        self.assertEqual(100, self.evaluator.lang_score)

    def test_exact_match_eng_lug(self):
        path = self.create_file(
            "exact_match_eng_lug.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "lug",
                "data": [{"word": "are", "prediction": "anganu"}]
            })
        )
        fake_lexicon = [
            {"source_word": "are", "target_translations": ["anganu"], "src_lang": "eng", "tgt_lang": "lug"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["anganu"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["anganu"]["exact_match"]))
        self.assertEqual(100, self.evaluator.lang_score)

    def test_inflection_spa_eng_2(self):
        path = self.create_file(
            "inflection_spa_eng.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "banco", "prediction": "schools of fish"}]
            })
        )
        fake_lexicon = [
            {"source_word": "banco", "target_translations": ["school of fish", "bank", "bench", "sandbank", "shoal"], "src_lang": "spa", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["banco"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["banco"]["inflection"]))
        self.assertEqual(100, self.evaluator.lang_score)

    def test_substring_spa_eng_1(self):
        path = self.create_file(
            "substring_spa_eng_1.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "carta", "prediction": "a letter from the kind"}]
            })
        )
        fake_lexicon = [
            {"source_word": "carta", "target_translations": ["menu", "letter"], "src_lang": "spa", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["carta"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["carta"]["substring"]))
        self.assertEqual(100, self.evaluator.lang_score)
    
    def test_substring_ayr_eng(self):
        path = self.create_file(
            "substring_ayr_eng.json",
            json.dumps({
                "src_lang": "ayr",
                "tgt_lang": "eng",
                "data": [{"word": "jacha", "prediction": "to stumble, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall on knees, fall"}]
            })
        )
        fake_lexicon = [
            {"source_word": "jacha", "target_translations": ["fall on knees"], "src_lang": "ayr", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["jacha"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["jacha"]["substring"]))
        self.assertEqual(100, self.evaluator.lang_score)
    
    def test_substring_zsm_eng(self):
        path = self.create_file(
            "substring_zsm_eng.json",
            json.dumps({
                "src_lang": "zsm",
                "tgt_lang": "eng",
                "data": [
                    {"word": "tetapi", "prediction": "but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but, but,"}
                ]
            })
        )
        fake_lexicon = [
            {"source_word": "tetapi", "target_translations": ["but"], "src_lang": "zsm", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["tetapi"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["tetapi"]["substring"]))
        self.assertEqual(100, self.evaluator.lang_score)
    
    def test_inflection_within_substring_spa_eng(self):
        path = self.create_file(
            "inflection_within_substring_spa_eng.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "tienda", "prediction": "the walmart store"}]
            })
        )
        fake_lexicon = [
            {"source_word": "tienda", "target_translations": ["stores", "tent"], "src_lang": "spa", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["tienda"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["tienda"]["inflection_within_substring"]))
        self.assertEqual(100, self.evaluator.lang_score)

    def test_synonym_spa_eng(self):
        path = self.create_file(
            "synonym_spa_eng.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "niña", "prediction": "miss"}]
            })
        )
        fake_lexicon = [
            {"source_word": "niña", "target_translations": ["girl"], "src_lang": "spa", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["niña"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["niña"]["synonym"]))
        self.assertEqual(100, self.evaluator.lang_score)

    def test_synonym_sun_eng(self):
        path = self.create_file(
            "synonym_sun_eng.json",
            json.dumps({
                "src_lang": "sun",
                "tgt_lang": "eng",
                "data": [{"word": "keukeuh", "prediction": "sincere"}]
            })
        )
        fake_lexicon = [
            {"source_word": "keukeuh", "target_translations": ["solemn"], "src_lang": "sun", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["keukeuh"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["keukeuh"]["synonym"]))
        self.assertEqual(100, self.evaluator.lang_score)

    def test_synonym_in_substring_spa_eng(self):
        path = self.create_file(
            "synonym_in_substring_spa_eng.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "viento", "prediction": "a fart"}]
            })
        )
        fake_lexicon = [
            {"source_word": "viento", "target_translations": ["wind"], "src_lang": "spa", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["viento"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["viento"]["synonym"]))
        self.assertEqual(100, self.evaluator.lang_score)

    def test_echo_eng_awa(self):
        path = self.create_file(
            "echo_eng_awa.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "awa",
                "data": [{"word": "for", "prediction": "for"}]
            })
        )
        fake_lexicon = [
            {"source_word": "for", "target_translations": ["खातिर"], "src_lang": "eng", "tgt_lang": "awa"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["खातिर"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["खातिर"]["echo"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_echo_eng_ceb_1(self):
        path = self.create_file(
            "echo_eng_ceb_1.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "ceb",
                "data": [{"word": "or", "prediction": "or"}]
            })
        )
        fake_lexicon = [
            {"source_word": "or", "target_translations": ["ug", "og"], "src_lang": "eng", "tgt_lang": "ceb"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ug"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ug"]["echo"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["og"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["og"]["echo"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_echo_eng_ceb_2(self):
        path = self.create_file(
            "echo_eng_ceb_2.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "ceb",
                "data": [{"word": "in", "prediction": "in"}]
            })
        )
        fake_lexicon = [
            {"source_word": "in", "target_translations": ["sa"], "src_lang": "eng", "tgt_lang": "ceb"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sa"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sa"]["echo"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_outputted_in_source_language_spa_eng_1(self):
        path = self.create_file(
            "outputted_in_source_language_spa_eng_1.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "peces", "prediction": "gatos"}]
            })
        )
        fake_lexicon = [
            {"source_word": "peces", "target_translations": ["fish"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "gatos", "target_translations": ["cat"], "src_lang": "spa", "tgt_lang": "eng"},
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["peces"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["peces"]["outputted_in_source_language"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_outputted_in_source_language_spa_eng_2(self):
        path = self.create_file(
            "outputted_in_source_language_spa_eng_2.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "banco", "prediction": "plateados"}]
            })
        )
        fake_lexicon = [
            {"source_word": "banco", "target_translations": ["school of fish", "bank", "bench", "sandbank", "shoal"], "src_lang": "spa", "tgt_lang": "eng"},
            {"source_word": "plateados", "target_translations": ["silver", "silvery"], "src_lang": "spa", "tgt_lang": "eng"},
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["banco"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["banco"]["outputted_in_source_language"]))
        self.assertEqual(0, self.evaluator.lang_score)
    
    def test_gibberish_ayr_eng(self):
        path = self.create_file(
            "gibberish_ayr_eng.json",
            json.dumps({
                "src_lang": "ayr",
                "tgt_lang": "eng",
                "data": [{"word": "uka", "prediction": "wakicht'atar"}]
            })
        )
        fake_lexicon = [
            {"source_word": "uka", "target_translations": ["that"], "src_lang": "ayr", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["uka"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["uka"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_jav_eng_1(self):
        path = self.create_file(
            "gibberish_jav_eng_1.json",
            json.dumps({
                "src_lang": "jav",
                "tgt_lang": "eng",
                "data": [{"word": "sasi", "prediction": "八月"}]
            })
        )
        fake_lexicon = [
            {"source_word": "sasi", "target_translations": ["month"], "src_lang": "jav", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sasi"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sasi"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_jav_eng_2(self):
        path = self.create_file(
            "gibberish_jav_eng_2.json",
            json.dumps({
                "src_lang": "jav",
                "tgt_lang": "eng",
                "data": [{"word": "ora", "prediction": "ughhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh"}]
            })
        )
        fake_lexicon = [
            {"source_word": "ora", "target_translations": ["no", "not"], "src_lang": "jav", "tgt_lang": "eng'"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ora"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ora"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_spa_eng(self):
        path = self.create_file(
            "gibberish_spa_eng.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "estrellas", "prediction": "earth"}]
            })
        )
        fake_lexicon = [
            {"source_word": "estrellas", "target_translations": ["stars", "celebrities"]}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["estrellas"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["estrellas"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_sun_eng(self):
        path = self.create_file(
            "gibberish_sun_eng.json",
            json.dumps({
                "src_lang": "sun",
                "tgt_lang": "eng",
                "data": [{"word": "sieun", "prediction": "mengejutkan"}]
            })
        )
        fake_lexicon = [
            {"source_word": "sieun", "target_translations": ["afraid", "scared", "helpful"]}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sieun"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["sieun"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_zsm_eng_1(self):
        path = self.create_file(
            "gibberish_zsm_eng_1.json",
            json.dumps({
                "src_lang": "zsm",
                "tgt_lang": "eng",
                "data": [{"word": "Bahasa", "prediction": "语言"}]
            })
        )
        fake_lexicon = [
            {"source_word": "bahasa", "target_translations": ["language", "speech"], "src_lang": "zsm", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["bahasa"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["bahasa"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_zsm_eng_2(self):
        path = self.create_file(
            "gibberish_zsm_eng_2.json",
            json.dumps({
                "src_lang": "zsm",
                "tgt_lang": "eng",
                "data": [{"word": "Ibu", "prediction": "母语 Language spoken or written by the majority of speakers or writers of a particular language."}]
            })
        )
        fake_lexicon = [
            {"source_word": "ibu", "target_translations": ["mother"], "src_lang": "zsm", "tgt_lang": "eng"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ibu"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ibu"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_eng_ayr(self):
        path = self.create_file(
            "gibberish_eng_ayr.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "ayr",
                "data": [{"word": "Many", "prediction": "'Many' in the above sentence refers to the number of people who think about them as dinosaurs."}]
            })
        )
        fake_lexicon = [
            {"source_word": "many", "target_translations": ["walja"], "src_lang": "eng", "tgt_lang": "ayr"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["walja"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["walja"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)
    
    def test_gibberish_eng_ceb(self):
        path = self.create_file(
            "gibberish_eng_ceb.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "ceb",
                "data": [{"word": "from", "prediction": "a. origin"}]
            })
        )
        fake_lexicon = [
            {"source_word": "from", "target_translations": ["gikan"], "src_lang": "eng", "tgt_lang": "ceb"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gikan"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gikan"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_eng_cjk(self):
        path = self.create_file(
            "gibberish_eng_cjk.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "cjk",
                "data": [{"word": "we", "prediction": "us"}]
            })
        )
        fake_lexicon = [
            {"source_word": "we", "target_translations": ["tuwé", "twé"], "src_lang": "eng", "tgt_lang": "cjk"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["tuwé"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["tuwé"]["gibberish"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["twé"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["twé"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_eng_lug(self):
        path = self.create_file(
            "gibberish_eng_lug.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "lug",
                "data": [{"word": "Children", "prediction": "'Children' in 'Children are placed in Foster Care for a wide variety of reasons that range from neglect, to abuse, and even to extortion.'"}]
            })
        )
        fake_lexicon = [
            {"source_word": "children", "target_translations": ["abaana"], "src_lang": "eng", "tgt_lang": "lug"}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["abaana"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["abaana"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_equivalence(self):
        path = self.create_file(
            "equivalence.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [
                    {"word": "gato", "prediction": "cat"}, # exact match
                    {"word": "peces", "prediction": "peces"}, # echo
                    {"word": "banco", "prediction": "plateados"}, # outputted in source language
                    {"word": "banco", "prediction": "schools of fish"}, # inflection
                    {"word": "niña", "prediction": "miss"} # synonynm
                ]
            })
        )
        fake_lexicon = [
            {"source_word": "gato", "target_translations": ["sneaky person", "cat"]},
            {"source_word": "carta", "target_translations": ["menu", "letter"]},
            {"source_word": "peces", "target_translations": ["fish"]},
            {"source_word": "banco", "target_translations": ["school of fish", "bank", "bench", "sandbank", "shoal"]}, 
            {"source_word": "plateados", "target_translations": ["silver", "silvery"]},
            {"source_word": "tienda", "target_translations": ["stores", "tent"]},
            {"source_word": "niña", "target_translations": ["girl"]},
            {"source_word": "pelota", "target_translations": ["ball"]}
        ]
        with patch.object(self.evaluator.loader, "get_omnis_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gato"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["gato"]["exact_match"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["peces"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["peces"]["echo"]))
        self.assertEqual(2, len(self.evaluator.xword_class_pred["banco"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["banco"]["outputted_in_source_language"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["banco"]["inflection"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["niña"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["niña"]["synonym"]))
        self.assertEqual(62.5, self.evaluator.lang_score)
