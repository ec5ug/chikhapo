from .base import BaseTaskFeeder
from chikhapo.utils.languages import convert_iso_to_name, get_direction_of_lang_pair, get_language_from_pair

class WordTranslationFeeder(BaseTaskFeeder):
    """
    for task Word Translation
    
    method: get_data_for_lang_pair
        returns a list of words in the given a language pair
        if lite==True, (deterministically) returns 300 elements
    method: get_prompts_for_lang_pair
        returns a list of prompts given a model name
        if lite==True, returns prompts from the (deterministically) random subset of words from get_data_for_lang_pair
    """

    def get_data_for_lang_pair(self, lang_pair, lite=True):
        list_of_words = []
        lexicon = self.loader.get_omnis_lexicon_subset(lang_pair)
        for entry in lexicon:
            list_of_words.append(entry["source_word"])
        if lite:
            list_of_words = self.get_random_sample(list_of_words)
        return list_of_words

    def get_prompts_for_lang_pair(self, model_name, lang_pair, lite=True):
        words = self.get_data_for_lang_pair(lang_pair, lite)
        prompts = []
        DIRECTION = get_direction_of_lang_pair(lang_pair)
        iso = get_language_from_pair(lang_pair)
        lang_name = convert_iso_to_name(iso)
        
        for word in words:
            if DIRECTION == "X_to_eng":
                if model_name == "aya-23-8b" or model_name == "falcon" or model_name == "llama":
                    prompt = f"Translate the following word from {lang_name} to English. Respond with a single word.\nWord: {word}\nTranslation: "
                elif model_name == "aya-101" or model_name == "bloom":
                    prompt = f"Translate the following text from {lang_name} to English: {word}."
                elif model_name == "gemma":
                    prompt = f"Translate '{word}' from {lang_name} into English. Respond in one word."
                else:
                    raise Exception(f"Model {model_name} was not tested in this study. No prompts are available")
            elif DIRECTION == "eng_to_X":
                if model_name == "aya-23-8b" or model_name == "falcon" or model_name == "llama":
                    prompt = f"Translate the following word from English to {lang_name}. Respond with a single word.\nWord: {word}\nTranslation: "
                elif model_name == "aya-101" or model_name == "bloom":
                    prompt = f"Translate the following text from English to {lang_name}: {word}."
                elif model_name == "gemma":
                    prompt = f"Translate '{word}' from English to {lang_name}. Answer in one word:"
                else:
                    raise Exception(f"Model {model_name} was not tested in this study. No prompts are available")
            prompts.append(prompt)
        return prompts