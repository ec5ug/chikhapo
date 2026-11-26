import re
from collections import defaultdict
from .base import BaseTaskFeeder
from chikhapo.utils.constants import DIRECTIONS
from chikhapo.utils.languages import convert_iso_to_name, get_direction_of_lang_pair, get_language_from_pair, get_language_pair
from chikhapo.utils.parsing import convert_list_of_entries_to_dictionary

class WordTranslationWithContextFeeder(BaseTaskFeeder):
    """
    The word translation with context task feeder returns a sentence containing the word to 
    be translated, the word to be translated, and the correct target translations.
    """
    def get_lang_pairs(self, DIRECTION=None):
        if DIRECTION is not None and DIRECTION not in DIRECTIONS:
            raise ValueError('DIRECTION should be None, "X_to_eng", or "eng_to_X"')
    
        chikhapo_isos = set(self.loader.get_chikhapo_lexicon_subset_names())
        iso_to_scripts = defaultdict(list)
        for iso_script in self.loader.get_glotlid_subset_names():
            iso_to_scripts[iso_script.split("_")[0]].append(iso_script)
    
        results = []
        for iso in sorted(chikhapo_isos & iso_to_scripts.keys()):
            for iso_script in sorted(iso_to_scripts[iso]):
                if DIRECTION in ("X_to_eng", None):
                    results.append(f"{iso_script}_eng")
                if DIRECTION in ("eng_to_X", None):
                    results.append(f"eng_{iso_script}")
        return results

    def get_data_for_lang_pair(self, iso_script_pair, lite=True):
        words_sentences_translations = {}
        lang_script = get_language_from_pair(iso_script_pair)
        text = self.loader.get_glotlid_subset(lang_script)
        
        # deriving the language pair for the lexicon
        DIRECTION = get_direction_of_lang_pair(iso_script_pair)
        iso = lang_script.split("_")[0]
        list_of_entries = self.loader.get_chikhapo_lexicon_subset(name=iso, split=DIRECTION)
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
    