from .base import BaseTaskFeeder
from chikhapo.utils.languages import convert_iso_to_name, get_direction_of_lang_pair, get_language_from_pair

class WordTranslationFeeder(BaseTaskFeeder):
    """
    The Word Translation task feeder returns a word in the source dictionary along with 
    verified target translations. Prompts are built from this data.
    """
    def get_lang_pairs(self, DIRECTION=None):
        omnis_subset_names = self.loader.get_omnis_lexicon_subset_names()
        omnis_subset_names.remove("all_eng")
        omnis_subset_names.remove("eng_all")
        if DIRECTION is None:
            return omnis_subset_names
        elif DIRECTION=="X_to_eng":
            return [c for c in omnis_subset_names if c.endswith('eng')]
        elif DIRECTION=="eng_to_X":
            return [c for c in omnis_subset_names if c.startswith('eng')]
        else:
            raise Exception("An invalid directon was specified. It should be None, \"X_to_eng\", or \"eng_to_X\"")

    def get_data_for_lang_pair(self, lang_pair, lite=True):
        words_tanslations = {}
        lexicon = self.loader.get_omnis_lexicon_subset(lang_pair)
        for entry in lexicon:
            k = entry["source_word"]
            v = entry["target_translations"]
            words_tanslations[k] = v
        if lite:
            words_tanslations = self.get_random_sample(words_tanslations)
        return words_tanslations

    def get_prompts_for_lang_pair(self, lang_pair, lite=True):
        words = self.get_data_for_lang_pair(lang_pair, lite)
        prompts = []
        DIRECTION = get_direction_of_lang_pair(lang_pair)
        iso = get_language_from_pair(lang_pair)
        lang_name = convert_iso_to_name(iso)
        
        for word in words:
            if DIRECTION == "X_to_eng":
                prompt = f"Translate the following word from {lang_name} to English. Respond with a single word.\nWord: {word}\nTranslation: "
            elif DIRECTION == "eng_to_X":
                prompt = f"Translate the following word from English to {lang_name}. Respond with a single word.\nWord: {word}\nTranslation: "
            prompts.append(prompt)
        return prompts
    