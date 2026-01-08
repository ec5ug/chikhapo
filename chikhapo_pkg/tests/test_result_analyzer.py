import unittest
from unittest.mock import Mock, patch
from chikhapo import ResultAnalyzer
from pathlib import Path
import os
import tempfile
import shutil
import pprint
import warnings

class TestResultAnalyzer(unittest.TestCase):
    def setUp(self):
        self.result_analyzer = None
        self.mocked_path_data = {
            # Indo-European
            "spa_eng.json": {"src_lang": "spa", "tgt_lang": "eng", "lang_score": 92},
            "deu_eng.json": {"src_lang": "deu", "tgt_lang": "eng", "lang_score": 90},
            "fra_eng.json": {"src_lang": "fra", "tgt_lang": "eng", "lang_score": 81},
            # Afro-Asiatic
            "arb_eng.json": {"src_lang": "arb", "tgt_lang": "eng", "lang_score": 72},
            "amh_eng.json": {"src_lang": "amh", "tgt_lang": "eng", "lang_score": 62},
            "heb_eng.json": {"src_lang": "heb", "tgt_lang": "eng", "lang_score": 76},
            # Atlantic-Congo
            "swh_eng.json": {"src_lang": "swh", "tgt_lang": "eng", "lang_score": 75},
            "wol_eng.json": {"src_lang": "wol", "tgt_lang": "eng", "lang_score": 40},
            "fuc_eng.json": {"src_lang": "fuc", "tgt_lang": "eng", "lang_score": 28}
        }
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def mock_evaluate(self, filepath):
        filename = os.path.basename(filepath)
        mocked_file_data = self.mocked_path_data[filename]
        self.result_analyzer.evaluator.src_lang = mocked_file_data["src_lang"]
        self.result_analyzer.evaluator.tgt_lang = mocked_file_data["tgt_lang"]
        self.result_analyzer.evaluator.lang_score = mocked_file_data["lang_score"] 

    def test_get_results_avg_stddev_by_language(self):
        self.result_analyzer = ResultAnalyzer("word_translation")
        
        with patch('os.listdir', return_value=list(self.mocked_path_data.keys())), \
            patch.object(self.result_analyzer.evaluator, "evaluate", side_effect=self.mock_evaluate):
            self.result_analyzer.get_results_by_language(self.temp_dir)

        self.assertEqual(9, len(self.result_analyzer.results_by_language))
        self.assertEqual(92, self.result_analyzer.results_by_language["spa"])
        self.assertEqual(90, self.result_analyzer.results_by_language["deu"])
        self.assertEqual(81, self.result_analyzer.results_by_language["fra"])
        self.assertEqual(72, self.result_analyzer.results_by_language["arb"])
        self.assertEqual(62, self.result_analyzer.results_by_language["amh"])
        self.assertEqual(76, self.result_analyzer.results_by_language["heb"])
        self.assertEqual(75, self.result_analyzer.results_by_language["swh"])
        self.assertEqual(40, self.result_analyzer.results_by_language["wol"])
        self.assertEqual(28, self.result_analyzer.results_by_language["fuc"])
        avg = self.result_analyzer.get_language_score_average()
        self.assertAlmostEqual(68.44, avg, places=2)
        std_dev = self.result_analyzer.get_language_score_standard_deviation()
        self.assertAlmostEqual(21.726, std_dev, places=2)

    def test_get_results_avg_stddev_by_language_family(self):
        self.result_analyzer = ResultAnalyzer("word_translation")
        
        with patch('os.listdir', return_value=list(self.mocked_path_data.keys())), \
            patch.object(self.result_analyzer.evaluator, "evaluate", side_effect=self.mock_evaluate):
            self.result_analyzer.get_results_by_language(self.temp_dir)
            self.result_analyzer.get_results_by_language_family()
        self.assertEqual(3, len(self.result_analyzer.results_by_language_family))
        self.assertEqual(set([92, 90, 81]), set(self.result_analyzer.results_by_language_family["Indo-European"]["scores"]))
        self.assertAlmostEqual(87.67, self.result_analyzer.results_by_language_family["Indo-European"]["avg"], places=2)
        self.assertAlmostEqual(5.86, self.result_analyzer.results_by_language_family["Indo-European"]["std_dev"], places=2)
        self.assertEqual(set([72, 62, 76]), set(self.result_analyzer.results_by_language_family["Afro-Asiatic"]["scores"]))
        self.assertEqual(70, self.result_analyzer.results_by_language_family["Afro-Asiatic"]["avg"])
        self.assertAlmostEqual(7.21, self.result_analyzer.results_by_language_family["Afro-Asiatic"]["std_dev"], places=2)
        self.assertEqual(set([75, 40, 28]), set(self.result_analyzer.results_by_language_family["Atlantic-Congo"]["scores"]))
        self.assertAlmostEqual(47.67, self.result_analyzer.results_by_language_family["Atlantic-Congo"]["avg"], places=2)
        self.assertAlmostEqual(24.42, self.result_analyzer.results_by_language_family["Atlantic-Congo"]["std_dev"], places=2)
    
    def test_system_word_translation(self):
        """
        The test data here comes from ChatGPT. The resulting xword_class_pred should 
        be non-empty and lang_score non-negative. Some results should be incorrect
        so we can expect some variation in the language scores.

        The languages evaluated on include:
        - Amharic (amh)     -> Afro-Asiatic
        - Armenian (hye)    -> Indo-European
        - Bashkir (bak)     -> Turkic
        - Dzongkha (dzo)    -> Sino-Tibetan
        - Hausa (hau)       -> Afro-Asiatic
        - Irish (gle)       -> Indo-European
        - Lao (lao)         -> Tai-Kadai
        - Thai (tha)        -> Tai-Kadai
        - Uyghur (uig)      -> Turkic
        - Xitsonga (tso)    -> Atlantic-Congo
        - Yoruba (yor)      -> Atlantic-Congo
        """
        self.result_analyzer = ResultAnalyzer("word_translation")
        data_dir = Path(__file__).resolve().parent / "raw_test_data" / "word_translation"
        self.result_analyzer.get_results_by_language(str(data_dir))
        # print(pprint.pformat(self.result_analyzer.results_by_language))
        # print(self.result_analyzer.get_language_score_average())
        # print(self.result_analyzer.get_language_score_standard_deviation())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.result_analyzer.get_results_by_language_family()
            relevant_warnings = [warning for warning in w if "You need at least two to calculate the standard deviation." in str(warning.message)]
            self.assertEqual(1, len(relevant_warnings))
        # print(pprint.pformat(self.result_analyzer.results_by_language_family))
        self.assertEqual(set(["amh", "hye", "bak", "dzo", "hau", "gle", "lao", "tha", "uig", "tso", "yor"]), set(self.result_analyzer.results_by_language.keys()))
        for score in self.result_analyzer.results_by_language.values():
            self.assertTrue(score > 0 and score <= 100)
        overall_avg = self.result_analyzer.get_language_score_average()
        self.assertTrue(overall_avg > 0 and overall_avg <= 100)
        overall_std_dev = self.result_analyzer.get_language_score_standard_deviation()
        self.assertTrue(overall_std_dev > 0)
        self.assertEqual(set(["Afro-Asiatic", "Indo-European", "Turkic", "Sino-Tibetan", "Tai-Kadai", "Atlantic-Congo"]), set(self.result_analyzer.results_by_language_family.keys()))
        for lang_fam in self.result_analyzer.results_by_language_family:
            avg = self.result_analyzer.results_by_language_family[lang_fam]["avg"]
            std_dev = self.result_analyzer.results_by_language_family[lang_fam]["std_dev"]
            self.assertTrue(avg > 0 and avg <= 100)
            if lang_fam != "Sino-Tibetan":
                self.assertTrue(std_dev > 0)
        self.assertTrue(-1, self.result_analyzer.results_by_language_family["Sino-Tibetan"]["std_dev"])

    def test_system_word_translation_with_context(self):
        self.result_analyzer = ResultAnalyzer("word_translation_with_context")
        data_dir = Path(__file__).resolve().parent / "raw_test_data" / "word_translation_with_context"
        self.result_analyzer.get_results_by_language(str(data_dir))
    #     # print(pprint.pformat(self.result_analyzer.results_by_language))
    #     # print(self.result_analyzer.get_language_score_average())
    #     # print(self.result_analyzer.get_language_score_standard_deviation())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.result_analyzer.get_results_by_language_family()
            relevant_warnings = [warning for warning in w if "You need at least two to calculate the standard deviation." in str(warning.message)]
            self.assertEqual(1, len(relevant_warnings))
        # print(pprint.pformat(self.result_analyzer.results_by_language_family))
        self.assertEqual(set(["amh", "hye", "bak", "dzo", "hau", "gle", "lao", "tha", "uig", "tso", "yor"]), set(self.result_analyzer.results_by_language.keys()))
        for score in self.result_analyzer.results_by_language.values():
            self.assertTrue(score > 0 and score <= 100)
        overall_avg = self.result_analyzer.get_language_score_average()
        self.assertTrue(overall_avg > 0 and overall_avg <= 100)
        overall_std_dev = self.result_analyzer.get_language_score_standard_deviation()
        self.assertTrue(overall_std_dev > 0)
        self.assertEqual(set(["Afro-Asiatic", "Indo-European", "Turkic", "Sino-Tibetan", "Tai-Kadai", "Atlantic-Congo"]), set(self.result_analyzer.results_by_language_family.keys()))
        for lang_fam in self.result_analyzer.results_by_language_family:
            avg = self.result_analyzer.results_by_language_family[lang_fam]["avg"]
            std_dev = self.result_analyzer.results_by_language_family[lang_fam]["std_dev"]
            self.assertTrue(avg > 0 and avg <= 100)
            if lang_fam != "Sino-Tibetan":
                self.assertTrue(std_dev > 0)
        self.assertTrue(-1, self.result_analyzer.results_by_language_family["Sino-Tibetan"]["std_dev"])

    def test_system_translation_conditioned_language_modeling(self):
        """
        Dzongzha is missing
        """
        self.result_analyzer = ResultAnalyzer("translation_conditioned_language_modeling")
        data_dir = Path(__file__).resolve().parent / "raw_test_data" / "translation_conditioned_language_modeling"
        self.result_analyzer.get_results_by_language(str(data_dir))
        self.result_analyzer.get_results_by_language_family()
        self.assertEqual(set(["amh", "hye", "bak", "hau", "gle", "lao", "tha", "uig", "tso", "yor"]), set(self.result_analyzer.results_by_language.keys()))
        for score in self.result_analyzer.results_by_language.values():
            self.assertTrue(score > 0 and score <= 100)
        overall_avg = self.result_analyzer.get_language_score_average()
        self.assertTrue(overall_avg > 0 and overall_avg <= 100)
        overall_std_dev = self.result_analyzer.get_language_score_standard_deviation()
        self.assertTrue(overall_std_dev > 0)
        self.assertEqual(set(["Afro-Asiatic", "Indo-European", "Turkic", "Tai-Kadai", "Atlantic-Congo"]), set(self.result_analyzer.results_by_language_family.keys()))
        for lang_fam in self.result_analyzer.results_by_language_family:
            avg = self.result_analyzer.results_by_language_family[lang_fam]["avg"]
            std_dev = self.result_analyzer.results_by_language_family[lang_fam]["std_dev"]
            self.assertTrue(avg > 0 and avg <= 100)
            self.assertTrue(std_dev > 0)

    def test_system_bag_of_words_machine_translation(self):
        """
        Dzongkha, Laos, Thai is missing
        """
        self.result_analyzer = ResultAnalyzer("bag_of_words_machine_translation")
        data_dir = Path(__file__).resolve().parent / "raw_test_data" / "bag_of_words_machine_translation"
        self.result_analyzer.get_results_by_language(str(data_dir))
        self.result_analyzer.get_results_by_language_family()
        self.assertEqual(set(["amh", "hye", "bak", "hau", "gle", "uig", "tso", "yor"]), set(self.result_analyzer.results_by_language.keys()))
        for score in self.result_analyzer.results_by_language.values():
            self.assertTrue(score > 0 and score <= 100)
        overall_avg = self.result_analyzer.get_language_score_average()
        self.assertTrue(overall_avg > 0 and overall_avg <= 100)
        overall_std_dev = self.result_analyzer.get_language_score_standard_deviation()
        self.assertTrue(overall_std_dev > 0)
        self.assertEqual(set(["Afro-Asiatic", "Indo-European", "Turkic", "Atlantic-Congo"]), set(self.result_analyzer.results_by_language_family.keys()))
        for lang_fam in self.result_analyzer.results_by_language_family:
            avg = self.result_analyzer.results_by_language_family[lang_fam]["avg"]
            std_dev = self.result_analyzer.results_by_language_family[lang_fam]["std_dev"]
            self.assertTrue(avg > 0 and avg <= 100)
            self.assertTrue(std_dev > 0)
        