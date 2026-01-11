"""
Initializes all types of task feeders, including task-specific,
"""

from .base import BaseTaskFeeder
from .task_feeder_factory import TaskFeeder
from .word_translation import WordTranslationFeeder
from .word_translation_with_context import WordTranslationWithContextFeeder
from .translation_conditioned_language_modeling import TranslationedConditionedLanguageModelingTaskFeeder
from .bag_of_words_machine_translation import BagOfWordsMachineTranslationFeeder

__all__ = [
    "BaseTaskFeeder",
    "TaskFeeder",
    "WordTranslationFeeder",
    "WordTranslationWithContextFeeder",
    "TranslationedConditionedLanguageModelingTaskFeeder",
    "BagOfWordsMachineTranslationFeeder"
]
