import re
from .base import BaseTaskFeeder
from chikhapo.utils.languages import convert_iso_to_name, get_direction_of_lang_pair, get_language_from_pair, get_language_pair

class WordTranslationWithContextFeeder(BaseTaskFeeder):
    """Handles word translation with context tasks"""
    
    def get_data_for_lang_pair(self, iso_script_pair):
        list_of_words_sentences = []
        lang_script = get_language_from_pair(iso_script_pair)
        text = self.loader.get_glotlid_subset(lang_script)
        
        # deriving the language pair for the lexicon
        direction = get_direction_of_lang_pair(iso_script_pair)
        iso = lang_script.split("_")[0]
        iso_pair = get_language_pair(iso, direction)
        lexicon = self.loader.get_omnis_lexicon_subset(iso_pair)
        
        # start parsing the dataset
        lex_words = set()
        for entry in lexicon:
            lex_words.add(entry["source_word"])
        
        for entry in text:
            sentence = entry["text"]
            lowercased_sentence = sentence.lower()
            raw_words = re.split(r"[\s\u00B2\u00B3\u00B9\u2070-\u2079]+", lowercased_sentence)
            for raw_word in raw_words:
                if raw_word in lex_words:
                    list_of_words_sentences.append((raw_word, sentence))
        return list_of_words_sentences

    def get_prompts_for_lang_pair(self, model_name, lang_pair):
        list_of_word_sentences = self.get_data_for_lang_pair(lang_pair)
        DIRECTION = get_direction_of_lang_pair(lang_pair)
        iso = get_language_from_pair(lang_pair)
        lang_name = convert_iso_to_name(iso)
        prompts = []
        
        for (word, sentence) in list_of_word_sentences:
            if DIRECTION == "X_to_eng":
                if model_name == "aya-101":
                    prompt = f"What does '{word}' mean in English in the sentence '{sentence[:-1]}'? Meaning (one word): "
                elif model_name == "aya-23-8b" or model_name == "falcon":
                    prompt = f"In '{sentence[:-1]}', the word '{word}' means ____ in English."
                elif model_name == "bloom" or model_name == "llama":
                    prompt = f"Sentence: {sentence[:-1]}\nDefine '{word}' in one English word: "
                elif model_name == "gemma":
                    prompt = f"Sentence: {sentence[:-1]}\nEnglish definition of '{word}': "
                else:
                    raise Exception("Incorrect MODEL")
            elif DIRECTION == "eng_to_X":
                if model_name == "aya-101":
                    prompt = f"What does '{word}' mean in {lang_name} in the sentence '{sentence[:-1]}'? Meaning (one word): "
                elif model_name == "aya-23-8b" or model_name == "falcon" or model_name == "gemma" or model_name == "llama":
                    prompt = f"In '{sentence[:-1]}', the word '{word}' means ____ in {lang_name}."
                elif model_name == "bloom":
                    prompt = f"Define '{word}' in '{sentence[:-1]}' in {lang_name}: "
                else:
                    raise Exception("Incorrect MODEL")
            prompts.append(prompt)
        return prompts