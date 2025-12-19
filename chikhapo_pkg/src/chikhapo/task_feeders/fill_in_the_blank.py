from .base import BaseTaskFeeder
from chikhapo.utils.languages import get_direction_of_lang_pair, get_language_from_pair

class FillInTheBlankFeeder(BaseTaskFeeder):
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
        DIRECTION = get_direction_of_lang_pair(lang_pair)
        iso_script = get_language_from_pair(lang_pair)
        if DIRECTION == "X_to_eng":
            src_dataset = self.loader.get_flores_subset(iso_script, split="devtest")
            tgt_dataset = self.loader.get_flores_subset("eng_Latn", split="devtest")
        else: # DIRECTION == "eng_to_X"
            src_dataset = self.loader.get_flores_subset("eng_Latn", split="devtest")
            tgt_dataset = self.loader.get_flores_subset(iso_script, split="devtest")
        src_sentences = [sentence["text"] for sentence in src_dataset]
        tgt_sentences = [sentence["text"] for sentence in tgt_dataset]
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
