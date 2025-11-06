import re
from .base import BaseTaskFeeder
from chikhapo.utils.languages import convert_iso_to_name, get_direction_of_lang_pair, get_language_from_pair, get_language_pair
from chikhapo.utils.parsing import convert_list_of_entries_to_dictionary
class WordTranslationWithContextFeeder(BaseTaskFeeder):
    """
    for task Word Translation with Context
    """
    
    def get_lang_pairs(self, DIRECTION=None):
        omnis_subset_names = set(self.loader.get_omnis_lexicon_subset_names())
        glotlid_names = set(self.loader.get_glotlid_subset_names())
        omnis_and_glotlid_subset_names = omnis_subset_names.intersection(glotlid_names)
        if DIRECTION is None:
            return omnis_and_glotlid_subset_names
        elif DIRECTION=="X_to_eng":
            return [c for c in omnis_and_glotlid_subset_names if c.endswith('eng')]
        elif DIRECTION=="eng_to_X":
            return [c for c in omnis_and_glotlid_subset_names if c.endswith('eng')]
        else:
            raise Exception("An invalid directon was specified. It should be None, \"X_to_eng\", or \"eng_to_X\"")

    def get_data_for_lang_pair(self, iso_script_pair, lite=True):
        words_sentences_translations = {}
        lang_script = get_language_from_pair(iso_script_pair)
        text = self.loader.get_glotlid_subset(lang_script)
        
        # deriving the language pair for the lexicon
        direction = get_direction_of_lang_pair(iso_script_pair)
        iso = lang_script.split("_")[0]
        iso_pair = get_language_pair(iso, direction)
        list_of_entries = self.loader.get_omnis_lexicon_subset(iso_pair)
        lexicon = convert_list_of_entries_to_dictionary(list_of_entries)
        
        for entry in text:
            sentence = entry["text"]
            lowercased_sentence = sentence.lower()
            raw_words = re.split(r"[\s\u00B2\u00B3\u00B9\u2070-\u2079]+", lowercased_sentence)
            for raw_word in raw_words:
                if raw_word in lexicon:
                    if (raw_word, sentence) not in words_sentences_translations:
                        words_sentences_translations[(raw_word, sentence)] = lexicon[raw_word]
        if lite:
            words_sentences_translations = self.get_random_sample(words_sentences_translations)
        return words_sentences_translations

    def get_prompts_for_lang_pair(self, lang_pair, lite=True):
        list_of_word_sentences = self.get_data_for_lang_pair(lang_pair, lite)
        DIRECTION = get_direction_of_lang_pair(lang_pair)
        iso = get_language_from_pair(lang_pair)
        lang_name = convert_iso_to_name(iso)
        prompts = []
        
        for (word, sentence) in list_of_word_sentences:
            if DIRECTION == "X_to_eng":
                prompt = f"What does '{word}' mean in English in the sentence '{sentence[:-1]}'? Meaning (one word): "
            elif DIRECTION == "eng_to_X":
                prompt = f"What does '{word}' mean in {lang_name} in the sentence '{sentence[:-1]}'? Meaning (one word): "
            prompts.append(prompt)
        return prompts