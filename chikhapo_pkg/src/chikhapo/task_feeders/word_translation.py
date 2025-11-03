from .base import BaseTaskFeeder
from chikhapo.utils.languages import convert_iso_to_name, get_direction_of_lang_pair, get_language_from_pair

class WordTranslationFeeder(BaseTaskFeeder):
    """Handles word translation tasks"""
    
    def get_data_for_lang_pair(self, lang_pair):
        list_of_words = []
        lexicon = self.loader.get_omnis_lexicon_subset(lang_pair)
        for entry in lexicon:
            list_of_words.append(entry["source_word"])
        return list_of_words

    def get_prompts_for_lang_pair(self, model_name, lang_pair):
        words = self.get_data_for_lang_pair(lang_pair)
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