"""
Task Feeder uses the factory design pattern selecting the task feeder at runtime 
based on a given task name
"""

from .word_translation import WordTranslationFeeder
from .word_translation_with_context import WordTranslationWithContextFeeder
from .translation_conditioned_language_modeling import TranslationedConditionedLanguageModelingTaskFeeder
from .bag_of_words_machine_translation import BagOfWordsMachineTranslationFeeder

def TaskFeeder(task_name):
    task_map = {
        "word_translation": WordTranslationFeeder,
        "word_translation_with_context": WordTranslationWithContextFeeder,
        "translation_conditioned_language_modeling": TranslationedConditionedLanguageModelingTaskFeeder,
        "bag_of_words_machine_translation": BagOfWordsMachineTranslationFeeder
    }
    
    if task_name not in task_map:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(task_map.keys())}"
        )
    
    return task_map[task_name]()
