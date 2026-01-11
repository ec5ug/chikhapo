from .base import BaseTaskFeeder
from chikhapo.utils.languages import convert_iso_to_name, get_direction_of_lang_pair, get_language_from_pair

class TranslationedConditionedLanguageModelingTaskFeeder(BaseTaskFeeder):
    """
    The data sourced by the translation conditioned language modeling task feeder is a 
    dictionary containing the source sentence, the i-th word in the target translation 
    to be predicted, the target translation up the i-th index (non-inclusive), and the 
    i-th target (ground-truth) word to be predicted. Prompts are constructed from 
    sections of these data.
    """
    def get_lang_pairs(self, DIRECTION=None):
        flores_subset_names = self.loader.get_flores_subset_names()
        flores_subset_names.remove("eng_Latn")
        to_eng = []
        from_eng = []
        for name in flores_subset_names:
            to_eng.append(f"{name}_eng")
            from_eng.append(f"eng_{name}")
        if DIRECTION is None:
            return to_eng+from_eng
        elif DIRECTION=="X_to_eng":
            return to_eng
        elif DIRECTION=="eng_to_X":
            return from_eng
        else:
            raise Exception("An invalid directon was specified. It should be None, \"X_to_eng\", or \"eng_to_X\"")

    def get_data_for_lang_pair(self, lang_pair, lite=True):
        src_sentences, tgt_sentences = self.loader.get_flores_subset_src_tgt_sentences(lang_pair)
        srcSentence_wordIndex_truncatedTrunslation_nextWord = {}
        for src_sentence, tgt_sentence in zip(src_sentences, tgt_sentences):
            tgt_words = tgt_sentence.split()
            for i in range(len(tgt_words)):
                truncated_translation = " ".join(tgt_words[:i])
                next_word = tgt_words[i]
                srcSentence_wordIndex_truncatedTrunslation_nextWord[(src_sentence, i)] = {
                    "truncated_translation": truncated_translation, 
                    "next_word": next_word
                }
        if lite:
            srcSentence_wordIndex_truncatedTrunslation_nextWord = self.get_random_sample(srcSentence_wordIndex_truncatedTrunslation_nextWord)
        return srcSentence_wordIndex_truncatedTrunslation_nextWord

    def get_prompts_for_lang_pair(self, lang_pair, lite=True):
        srcSentence_wordIndex_truncatedTrunslation_nextWord = self.get_data_for_lang_pair(lang_pair=lang_pair, lite=lite)
        prompts = []
        DIRECTION = get_direction_of_lang_pair(lang_pair)
        iso_script = get_language_from_pair(lang_pair)
        iso = iso_script.split("_")[0]
        lang_name = convert_iso_to_name(iso)

        for record_key, record_value in srcSentence_wordIndex_truncatedTrunslation_nextWord.items():
            src_sentence, _ = record_key
            truncated_translation = record_value["truncated_translation"]
            if DIRECTION=="X_to_eng":
                prompts.append(
                    f"Translate the sentence into English:\n{lang_name}: {src_sentence}\nEnglish: {truncated_translation}"
                )
            else: # DIRECTION=="eng_to_X"
                prompts.append(
                    f"Translate the following text into {lang_name}.\nEnglish: {src_sentence}\n{lang_name}: {truncated_translation}"
                )
        return prompts
    