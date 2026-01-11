from .base import BaseTaskFeeder
from .translation_conditioned_language_modeling import TranslationedConditionedLanguageModelingTaskFeeder
from chikhapo.utils.languages import convert_iso_to_name, get_direction_of_lang_pair, get_language_from_pair

class BagOfWordsMachineTranslationFeeder(BaseTaskFeeder):
    """
    The task feeder for Bag-of-Words Machine Translation returns source and target 
    (ground-truth) translations
    """
    def get_lang_pairs(self, DIRECTION=None):
        return TranslationedConditionedLanguageModelingTaskFeeder().get_lang_pairs(DIRECTION)
    
    def get_data_for_lang_pair(self, lang_pair, lite=True):
        src_sentences, tgt_sentences = self.loader.get_flores_subset_src_tgt_sentences(lang_pair)
        srcSentence_tgtSentence = {}
        for src_sentence, tgt_sentence in zip(src_sentences, tgt_sentences):
            srcSentence_tgtSentence[src_sentence] = tgt_sentence
        if lite:
            return self.get_random_sample(srcSentence_tgtSentence)
        return srcSentence_tgtSentence
    
    def get_prompts_for_lang_pair(self, lang_pair, lite=True):
        srcSentence_tgtSentence = self.get_data_for_lang_pair(lang_pair=lang_pair, lite=lite)
        DIRECTION = get_direction_of_lang_pair(lang_pair)
        prompts = []
        for src_sentence in srcSentence_tgtSentence.keys():
            if DIRECTION=="X_to_eng":
                prompts.append(f"Translate into English: {src_sentence}")
            else: # DIRECTION=="eng_to_X"
                iso_script = get_language_from_pair(lang_pair)
                iso = iso_script.split("_")[0]
                language_name = convert_iso_to_name(iso)
                prompts.append(f"Translate into {language_name}: {src_sentence}")
        return prompts
    