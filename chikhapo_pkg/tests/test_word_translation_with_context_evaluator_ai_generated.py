import json
from unittest.mock import patch
from chikhapo import Evaluator
from .base_evaluator_test import BaseEvaluatorTest

real_data_path = "/share/data/speech-lang/users/echang/chikhapo/chikhapo_pkg/tests/raw_test_data/word_translation_with_context/appendix/parsed_responses/"
raw_item_path = "/share/data/speech-lang/users/echang/chikhapo/chikhapo_pkg/tests/raw_test_data/word_translation_with_context/appendix/item_responses/"

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
                "data": [
                    {"word": "gatos", "source_sentence": "Los gatos duermen en el sofá todo el día.", "prediction": "cat"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "gatos", "target_translations": ["cat"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "ancha", "source_sentence": "La calle es muy ancha en esta parte de la ciudad.", "prediction": "much"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "ancha", "target_translations": ["much"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "iyeu", "source_sentence": "Iyeu imah téh kacida gedéna.", "prediction": "they"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "iyeu", "target_translations": ["they"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "lain", "source_sentence": "Éta lain buku kuring.", "prediction": "other"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "lain", "target_translations": ["other"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "are", "source_sentence": "The children are playing outside.", "prediction": "anganu"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "are", "target_translations": ["anganu"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["anganu"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["anganu"]["exact_match"]))
        self.assertEqual(100, self.evaluator.lang_score)

    def test_inflection_spa_eng_2(self):
        path = self.create_file(
            "inflection_spa_eng_2.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [
                    {"word": "banco", "source_sentence": "Vimos un banco de peces cerca del arrecife.", "prediction": "schools of fish"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "banco", "target_translations": ["school of fish", "bank", "bench", "sandbank", "shoal"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "carta", "source_sentence": "Recibí una carta del rey ayer.", "prediction": "a letter from the kind"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "carta", "target_translations": ["menu", "letter"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "jacha", "source_sentence": "Jupax jacha uñjasaw sarawayi.", "prediction": "to stumble, fall on knees, " + "fall on knees, " * 22 + "fall"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "jacha", "target_translations": ["fall on knees"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                    {"word": "tetapi", "source_sentence": "Saya mahu pergi, tetapi hujan turun dengan lebat.", "prediction": "but, " * 64},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "tetapi", "target_translations": ["but"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "tienda", "source_sentence": "Compré pan en la tienda de la esquina.", "prediction": "the walmart store"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "tienda", "target_translations": ["stores", "tent"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "niña", "source_sentence": "La niña juega en el parque.", "prediction": "miss"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "niña", "target_translations": ["girl"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "keukeuh", "source_sentence": "Manéhna keukeuh hayang indit ka kota.", "prediction": "sincere"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "keukeuh", "target_translations": ["solemn"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "viento", "source_sentence": "El viento sopla muy fuerte esta noche.", "prediction": "a fart"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "viento", "target_translations": ["wind"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "for", "source_sentence": "This gift is for my mother.", "prediction": "for"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "for", "target_translations": ["खातिर"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "or", "source_sentence": "Do you want tea or coffee?", "prediction": "or"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "or", "target_translations": ["ug", "og"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "in", "source_sentence": "The keys are in the drawer.", "prediction": "in"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "in", "target_translations": ["sa"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "peces", "source_sentence": "Los peces nadan en el río.", "prediction": "gatos"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "peces", "target_translations": ["fish"]},
            {"source_word": "gatos", "target_translations": ["cat"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "banco", "source_sentence": "Un banco de peces plateados pasó junto al barco.", "prediction": "plateados"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "banco", "target_translations": ["school of fish", "bank", "bench", "sandbank", "shoal"]},
            {"source_word": "plateados", "target_translations": ["silver", "silvery"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["banco"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["banco"]["outputted_in_source_language"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_outputted_in_source_language_eng_ayr(self):
        path = self.create_file(
            "outputted_in_source_language_eng_ayr.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "ayr",
                "data": [
                    {"word": "Many", "source_sentence": "Many people think of them as dinosaurs.", "prediction": "'Many' in the above sentence refers to the number of people who think about them as dinosaurs."},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "many", "target_translations": ["walja"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["walja"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["walja"]["outputted_in_source_language"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_outputted_src_sentence_eng_lug(self):
        path = self.create_file(
            "outputted_src_sentence_eng_lug.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "lug",
                "data": [
                    {"word": "children", "source_sentence": "Children are placed in Foster Care for a wide variety of reasons that range from neglect, to abuse, and even to extortion.", "prediction": "'Children' in 'Children are placed in Foster Care for a wide variety of reasons that range from neglect, to abuse, and even to extortion.'"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "children", "target_translations": ["abaana"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["abaana"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["abaana"]["outputted_src_sentence"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_outputted_src_sentence_ban_eng(self):
        path = self.create_file(
            "outputted_src_sentence_ban_eng.json",
            json.dumps({
                "src_lang": "ban",
                "tgt_lang": "eng",
                "data": [
                    {"word": "ring", "source_sentence": "Ritatkala semeng tiang lunga ring peken sareng meme tiang jagi numbas sayur, bawang, miwah taluh anggen ngae ajengan.", "prediction": "the word 'ring' in 'Ritatkala semeng tiang lunga ring peken sareng meme tiang jagi numbas sayur, bawang, miwah taluh anggen ngae ajengan.'"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "ring", "target_translations": ["in", "at", "to"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ring"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ring"]["outputted_src_sentence"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_ayr_eng(self):
        path = self.create_file(
            "gibberish_ayr_eng.json",
            json.dumps({
                "src_lang": "ayr",
                "tgt_lang": "eng",
                "data": [
                    {"word": "uka", "source_sentence": "Uka utax wali jach'awa.", "prediction": "wakicht'atar"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "uka", "target_translations": ["that"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "sasi", "source_sentence": "Sasi ngarep aku arep lunga menyang Jakarta.", "prediction": "八月"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "sasi", "target_translations": ["month"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "ora", "source_sentence": "Aku ora ngerti apa sing kokomongké.", "prediction": "ugh" + "h" * 290},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "ora", "target_translations": ["no", "not"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "estrellas", "source_sentence": "Las estrellas brillan en el cielo.", "prediction": "earth"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "estrellas", "target_translations": ["stars", "celebrities"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "sieun", "source_sentence": "Budak éta sieun ku anjing.", "prediction": "mengejutkan"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "sieun", "target_translations": ["afraid", "scared", "helpful"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "Bahasa", "source_sentence": "Bahasa Melayu dituturkan di banyak negara.", "prediction": "语言"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "bahasa", "target_translations": ["language", "speech"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "Ibu", "source_sentence": "Ibu saya memasak nasi lemak setiap pagi.", "prediction": "母语 Language spoken or written by the majority of speakers or writers of a particular language."},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "ibu", "target_translations": ["mother"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ibu"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["ibu"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_gibberish_eng_ceb(self):
        path = self.create_file(
            "gibberish_eng_ceb.json",
            json.dumps({
                "src_lang": "eng",
                "tgt_lang": "ceb",
                "data": [
                    {"word": "from", "source_sentence": "She came from the market.", "prediction": "a. origin"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "from", "target_translations": ["gikan"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
                "data": [
                    {"word": "we", "source_sentence": "We walked to the river together.", "prediction": "us"},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "we", "target_translations": ["tuwé", "twé"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertEqual(1, len(self.evaluator.xword_class_pred["tuwé"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["tuwé"]["gibberish"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["twé"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["twé"]["gibberish"]))
        self.assertEqual(0, self.evaluator.lang_score)

    def test_equivalence(self):
        path = self.create_file(
            "equivalence.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [
                    {"word": "gato", "source_sentence": "El gato duerme en la cama.", "prediction": "cat"},
                    {"word": "peces", "source_sentence": "Los peces nadan en el río.", "prediction": "peces"},
                    {"word": "banco", "source_sentence": "Un banco de peces plateados pasó junto al barco.", "prediction": "plateados"},
                    {"word": "banco", "source_sentence": "Vimos un banco de peces cerca del arrecife.", "prediction": "schools of fish"},
                    {"word": "niña", "source_sentence": "La niña juega en el parque.", "prediction": "miss"},
                    {"word": "tienda", "source_sentence": "Compré pan en la tienda de la esquina.", "prediction": "Compré pan en la tienda de la esquina."},
                ],
            }, ensure_ascii=False),
        )
        fake_lexicon = [
            {"source_word": "gato", "target_translations": ["sneaky person", "cat"]},
            {"source_word": "carta", "target_translations": ["menu", "letter"]},
            {"source_word": "peces", "target_translations": ["fish"]},
            {"source_word": "banco", "target_translations": ["school of fish", "bank", "bench", "sandbank", "shoal"]},
            {"source_word": "plateados", "target_translations": ["silver", "silvery"]},
            {"source_word": "tienda", "target_translations": ["stores", "tent"]},
            {"source_word": "niña", "target_translations": ["girl"]},
            {"source_word": "pelota", "target_translations": ["ball"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
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
        self.assertEqual(1, len(self.evaluator.xword_class_pred["tienda"]))
        self.assertEqual(1, len(self.evaluator.xword_class_pred["tienda"]["outputted_src_sentence"]))
        self.assertEqual(50.0, self.evaluator.lang_score)

    def test_missing_source_sentence_raises(self):
        path = self.create_file(
            "missing_source_sentence.json",
            json.dumps({
                "src_lang": "spa",
                "tgt_lang": "eng",
                "data": [{"word": "gatos", "prediction": "cat"}],
            }),
        )
        fake_lexicon = [
            {"source_word": "gatos", "target_translations": ["cat"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
            with self.assertRaises(Exception):
                self.evaluator.evaluate(path)

    def test_word_with_punctuation_is_skipped(self):
        # lookup is lowercase-only (like the old ingestor), so "bahasa," does not match "bahasa"
        path = self.create_file(
            "word_with_punctuation_is_skipped.json",
            json.dumps({
                "src_lang": "zsm",
                "tgt_lang": "eng",
                "data": [{"word": "bahasa,", "source_sentence": "Saya suka bahasa, budaya dan makanan Malaysia.", "prediction": "language"}],
            }),
        )
        fake_lexicon = [
            {"source_word": "bahasa", "target_translations": ["language", "speech"]},
        ]
        with patch.object(self.evaluator.loader, "get_chikhapo_lexicon_subset", return_value=fake_lexicon):
            self.evaluator.evaluate(path)
        self.assertNotIn("bahasa", self.evaluator.xword_class_pred)
        self.assertNotIn("bahasa,", self.evaluator.xword_class_pred)
        self.assertEqual(0, self.evaluator.lang_score)
        