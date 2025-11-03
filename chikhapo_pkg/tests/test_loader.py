import unittest
from chikhapo import Loader


class TestLoader(unittest.TestCase):

    def setUp(self):
        self.loader = Loader()

    def test_get_flores_subset_names(self):
        flores_subset_names = self.loader.get_flores_subset_names()
        self.assertGreaterEqual(len(flores_subset_names), 200)

    def test_get_flores_subset(self):
        try:
            self.loader.get_flores_subset("spa_Latn", "devtest")
        except Exception as e:
            self.fail(f"Unexpected error in retrieving FLORES split: {e}")

    def test_get_glotlid_subset_names(self):
        glotlid_subset_names = self.loader.get_glotlid_subset_names()
        self.assertGreaterEqual(len(glotlid_subset_names), 1900)

    def test_get_glotlid_subset(self):
        try:
            self.loader.get_glotlid_subset("spa_Latn")
        except Exception as e:
            self.fail(f"Unexpected error in retrieving GLOTLID split: {e}")

    def test_get_omnis_subset_names(self):
        omnis_subset_names = self.loader.get_omnis_lexicon_subset_names()
        self.assertGreaterEqual(len(omnis_subset_names), 5000)

    def test_get_omnis_lexicon_subset(self):
        subset = self.loader.get_omnis_lexicon_subset("spa_eng")
        self.assertIn("source_word", subset[0])
        self.assertIn("target_translations", subset[0])
        self.assertIn("src_lang", subset[0])
        self.assertIn("tgt_lang", subset[0])


if __name__ == "__main__":
    unittest.main()
