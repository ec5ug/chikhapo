import sys
import pytest

from chikhapo import Loader

@pytest.fixture
def loader_instance():
    return Loader()

def test_get_flores_subset_names(loader_instance):
    flores_subset_names = loader_instance.get_flores_subset_names()
    assert len(flores_subset_names) >= 200

def test_get_flores_subset(loader_instance):
    try:
        loader_instance.get_flores_subset("spa_Latn", "devtest")
    except Exception:
        pytest.fail("Unexpected error in retrieving FLORES split")

def test_get_glotlid_subset_names(loader_instance):
    glotlid_subset_names = loader_instance.get_glotlid_subset_names()
    assert len(glotlid_subset_names) >= 1900

def test_get_glotlid_subset(loader_instance):
    try:
        loader_instance.get_glotlid_subset("spa_Latn")
    except Exception:
        pytest.fail("Unexpected error in retrieving GLOTLID split")

def test_get_omnis_subset_names(loader_instance):
    omnis_subset_names = loader_instance.get_omnis_lexicon_subset_names()
    assert len(omnis_subset_names) >= 5000

def test_get_omnis_lexicon_subset(loader_instance):
    s = loader_instance.get_omnis_lexicon_subset("spa_eng")
    assert "source_word" in s[0]
    assert "target_translations" in s[0]
    assert "src_lang" in s[0]
    assert "tgt_lang" in s[0]