"""
initializes all types of evaluators, including task-specific, parent class, 
and the factory
"""

from .evaluator_factory import Evaluator
from .word_translation import WordTranslationEvaluator
from .translation_conditioned_language_modeling import TranslationConditionedLanguageModelingEvaluator
from .bag_of_words_machine_translation import BagOfWordsMachineTranslationEvaluator
from .base import BaseEvaluator

__all__ = [
    "Evaluator",
    "WordTranslationEvaluator",
    "TranslationConditionedLanguageModelingEvaluator",
    "BagOfWordsMachineTranslationEvaluator",
    "BaseEvaluator"
]
