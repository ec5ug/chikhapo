from collections import defaultdict
from pathlib import Path
import re
from .loader import Loader
import sys
CURRENT_DIR = Path(__file__).resolve().parent
UTILS_DIR = CURRENT_DIR.parents[2] / "utils"
sys.path.append(str(UTILS_DIR))
from languages import convert_iso_to_name, get_direction_of_lang_pair, get_language_from_pair, get_language_pair

class Task_Feeder:
    def __init__(self):
        self.loader = Loader()

    def get_word_translation_data_for_lang_pair(self, lang_pair):
        list_of_words = []
        lexicon = self.loader.get_omnis_lexicon_subset(lang_pair)
        for entry in lexicon:
            list_of_words.append(entry["source_word"])
        return list_of_words

    # def get_word_translation_data(self):
    #     all_word_translation_lang_pairs = self.loader.get_omnis_lexicon_subset_names()
    #     langpair_words = defaultdict(list)
    #     for lang_pair in all_word_translation_lang_pairs:
    #         langpair_words[lang_pair] = self.get_word_translation_data_for_lang_pair(lang_pair)
    #         # for testing purposes
    #         # if len(langpair_words) > 5:
    #         #     return langpair_words
    #     return langpair_words
    
    # def get_word_translation_prompts(self, model_name): ### for all_eng

    def get_word_translation_prompts_for_lang_pair(self, model_name, lang_pair):
        words = self.get_word_translation_data_for_lang_pair(lang_pair)
        prompts = []
        # for langpair, words in langpair_words.items():
        DIRECTION = get_direction_of_lang_pair(lang_pair)
        iso = get_language_from_pair(lang_pair) # the non-english iso
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
    
    def get_word_translation_with_context_data_for_lang_pair(self, iso_script_pair):
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

    # def get_word_translation_with_context_data(self):
    #     omnis_lexicon_iso_pairs = self.loader.get_omnis_lexicon_subset_names()
    #     glotlid_iso_scripts = self.loader.get_glotlid_subset_names()
    #     omnis_glotlid_intersection = set()
    #     for iso_script in glotlid_iso_scripts:
    #         iso = iso_script.split("_")[0]
    #         if f"{iso}_eng" in omnis_lexicon_iso_pairs:
    #             omnis_glotlid_intersection.add(f"{iso_script}_eng")
    #         if f"eng_{iso}" in omnis_lexicon_iso_pairs:
    #             omnis_glotlid_intersection.add(f"eng_{iso_script}")
    #     omnis_glotlid_intersection = list(omnis_glotlid_intersection)
    #     langpair_word_sentence = defaultdict(list)
    #     for iso_script_pair in omnis_glotlid_intersection:
    #         langpair_word_sentence[iso_script_pair] = self.get_word_translation_with_context_data_for_lang_pair(iso_script_pair)
    #     return langpair_word_sentence

    def get_word_translation_with_context_prompts_for_lang_pair(self, model_name, lang_pair):
        list_of_word_sentences = self.get_word_translation_with_context_data_for_lang_pair(lang_pair)
        DIRECTION = get_direction_of_lang_pair(lang_pair)
        iso = get_language_from_pair(lang_pair) # the non-english iso
        lang_name = convert_iso_to_name(iso) # the full name of the non-english iso
        prompts = []
        for (word, sentence) in list_of_word_sentences:
            if DIRECTION == "X_to_eng":
                if model_name == "aya-101":
                    prompt = f"What does '{word}' mean in English in the sentence '{sentence[:-1]}'? Meaning (one word): "
                elif model_name == "aya-23-8b" or model_name == "falcon":
                    prompt = f"In '{sentence[:-1]}', the word '{word}' means ____ in English."
                elif model_name == "bloom" or model_name=="llama":
                    prompt = f"Sentence: {sentence[:-1]}\nDefine '{wod}' in one English word: "
                elif model_name == "gemma":
                    prompt = f"Sentence: {sentence[:-1]}\nEnglish definition of '{word}': "
                else:
                    raise Exception("Incorrect MODEL")
            elif DIRECTION == "eng_to_X":
                if model_name == "aya-101":
                    prompt = f"What does '{word}' mean in {lang_name} in the sentence '{sentence[:-1]}'? Meaning (one word): "
                elif model_name == "aya-23-8b" or model_name == "falcon" or model_name == "gemma" or MODEL=="llama":
                    prompt = f"In '{sentence[:-1]}', the word '{word}' means ____ in {lang_name}."
                elif model_name == "bloom":
                    prompt = f"Define '{word}' in '{sentence[:-1]}' in {lang_name}: "
                else:
                    raise Exception("Incorrect MODEL")
            prompts.append(prompt)
        return prompts

# if __name__=="__main__":
#     feeder = Task_Feeder()
#     print("Word Translation")
#     r = feeder.get_word_translation_data_for_lang_pair("aac_eng")
#     print("Translation data for lang pair")
#     for i in range(5):
#         print(r[i])
#     print()
#     r = feeder.get_word_translation_prompts_for_lang_pair("aya-101", "aac_eng")
#     print("Word Translation prompts")
#     for i in range(5):
#         print(r[i])
#     print("\n")
#     print("Word Translation with Context")
#     # 'eng_tab_Cyrl', 'eng_nde_Latn', 'eng_wal_Latn', 'eng_tbc_Latn', 'eng_naf_Latn', 'eng_mni_Latn', 'eng_lea_Latn', 'eng_trq_Latn', 'eng_sjo_Mong', 'eng_moa_Latn', 'eng_nhe_Latn', 'eng_csy_Latn', 'eng_blh_Latn', 'eng_tpa_Latn', 'eng_kle_Deva', 'eng_oci_Latn', 'eng_snd_Latn', 'eng_guk_Ethi', 'eng_rug_Latn', 'bvr_Latn_eng'
#     r = feeder.get_word_translation_with_context_data_for_lang_pair("anv_Latn_eng")
#     for i in range(5):
#         print(r[i])
#     r = feeder.get_word_translation_with_context_prompts_for_lang_pair("aya-101", "anv_Latn_eng")
#     for i in range(5):
#         print(r[i])