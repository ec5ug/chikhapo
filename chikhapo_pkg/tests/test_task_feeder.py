import sys
import pytest

from chikhapo import Task_Feeder

@pytest.fixture
def task_feeder_instance():
    return Task_Feeder()

def test_get_word_translation_data_for_lang_pair(task_feeder_instance):
    list_of_words = task_feeder_instance.get_word_translation_data_for_lang_pair("aac_eng")
    assert isinstance(list_of_words, list)
    assert len(list_of_words) > 0
    assert all(isinstance(w, str) for w in list_of_words)

def get_word_translation_with_context_data_for_lang_pair(task_feeder_instance):
    list_of_prompts = task_feeder_instance.get_word_translation_prompts("aya-101", "aac_eng")
    assert isinstance(list_of_prompts, list)
    assert len(list_of_prompts) > 0
    assert all(isinstance(w, str) for w in list_of_prompts)

def test_get_word_translation_with_context_data_for_lang_pair(task_feeder_instance):
    list_of_words_sentences = task_feeder_instance.get_word_translation_with_context_data_for_lang_pair("avn_Latn_eng")
    assert isinstance(list_of_words_sentences, list)
    assert len(list_of_words_sentences) > 0
    assert all(isinstance(w, tuple) for w in list_of_words_sentences)
    assert len(list_of_words_sentences[0]) == 2

def test_get_word_translation_with_context_prompts_for_lang_pair(task_feeder_instance):
    list_of_prompts = task_feeder_instance.get_word_translation_with_context_prompts_for_lang_pair("aya-101", "avn_Latn_eng")
    assert isinstance(list_of_prompts, list)
    assert len(list_of_prompts) > 0
    assert all(isinstance(w, str) for w in list_of_prompts)
