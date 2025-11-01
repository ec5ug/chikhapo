from collections import defaultdict
from loader import Loader
import sys
sys.path.append("../../../utils")
from languages import convert_iso_to_name, get_direction_of_lang_pair, get_language_from_pair

class Task_Feeder:
    def __init__(self):
        self.loader = Loader()

    def get_word_translation_data_for_lang_pair(self, lang_pair):
        list_of_words = []
        lexicon = self.loader.get_omnis_lexicon_subset(lang_pair)
        for entry in lexicon:
            list_of_words.append(entry["source"])
        return list_of_words

    def get_word_translation_data(self):
        all_word_translation_lang_pairs = self.loader.get_omnis_lexicon_subset_names()
        langpair_words = defaultdict(list)
        for lang_pair in all_word_translation_lang_pairs:
            langpair_words[lang_pair] = self.get_word_translation_data_for_lang_pair(lang_pair)
            # for testing purposes
            if len(langpair_words) > 5:
                return langpair_words
        return langpair_words
    
    def get_word_translation_prompts(self, model_name):
        langpair_words = self.get_word_translation_data()
        prompts = []
        for langpair, words in langpair_words.items():
            DIRECTION = get_direction_of_lang_pair(langpair)
            iso = get_language_from_pair(langpair) # the non-english iso
            lang_name = convert_iso_to_name(iso) # the full name of the non-english iso
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
    
if __name__=="__main__":
    feeder = Task_Feeder()
    r = feeder.get_word_translation_prompts("aya-101")
    for i in range(5):
        print(r[i])