from huggingface_hub import DatasetCard

from .base import BaseTaskFeeder
from chikhapo.utils.languages import convert_iso_to_name, get_direction_of_lang_pair, get_language_from_pair, get_language_pair
from chikhapo.utils.constants import DIRECTIONS

class WordTranslationFeeder(BaseTaskFeeder):
    """
    The Word Translation task feeder returns a word in the source dictionary along with 
    verified target translations. Prompts are built from this data.
    """
    def get_lang_pairs(self, DIRECTION=None):
        if DIRECTION is not None and DIRECTION not in DIRECTIONS:
            raise ValueError(f"DIRECTION must be None or one of {DIRECTIONS}, got '{DIRECTION}'")
        directions = list(DIRECTIONS) if DIRECTION is None else [DIRECTION]
        card = DatasetCard.load(self.loader.chikhapo_lexicons_hf_path, repo_type="dataset")
        return [
            get_language_pair(cfg["config_name"], d)
            for d in directions
            for cfg in card.data.configs
            if cfg["config_name"] != "all"
            and any(f["split"] == d for f in cfg["data_files"])
        ]

    def get_data_for_lang_pair(self, lang_pair, lite=True):
        words_translations = {}
        iso = get_language_from_pair(lang_pair)
        DIRECTION = get_direction_of_lang_pair(lang_pair)
        lexicon = self.loader.get_chikhapo_lexicon_subset(iso, DIRECTION)
        for entry in lexicon:
            source_word = entry["source_word"]
            target_translations = entry["target_translations"]
            lite_entry = entry["lite"]
            if lite and not lite_entry: # lite is turned on and lite_entry is false
                continue
            words_translations[source_word] = target_translations
        return words_translations

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
            else:
                raise Exception(f"No prompt available for DIRECTION {DIRECTION}")
            prompts.append(prompt)
        return prompts
    