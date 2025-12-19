from .base import BaseTaskFeeder
from .fill_in_the_blank import FillInTheBlankFeeder

class BagOfWordsMachineTranslationFeeder(BaseTaskFeeder):
    def get_lang_pairs(self, DIRECTION=None):
        return FillInTheBlankFeeder().get_lang_pairs(DIRECTION)
    
    def get_data_for_lang_pair(self, lang_pair, lite=True):
        src_sentences, tgt_sentences = self.loader.get_flores_subset_src_tgt_sentences(lang_pair)
        srcSentence_tgtSentence = {}
        for src_sentence, tgt_sentence in zip(src_sentences, tgt_sentences):
            srcSentence_tgtSentence[src_sentence] = tgt_sentence
        if lite:
            return self.get_random_sample(srcSentence_tgtSentence)
        return srcSentence_tgtSentence
        