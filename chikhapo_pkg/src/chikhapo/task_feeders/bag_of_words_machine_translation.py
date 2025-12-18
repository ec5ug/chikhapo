from .base import BaseTaskFeeder
from .fill_in_the_blank import FillInTheBlankFeeder

class BagOfWordsMachineTranslationFeeder(BaseTaskFeeder):
    def get_lang_pairs(self, DIRECTION=None):
        return FillInTheBlankFeeder().get_lang_pairs(DIRECTION)