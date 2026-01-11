from .word_translation import WordTranslationEvaluator
from .translation_conditioned_language_modeling import TranslationConditionedLanguageModelingEvaluator
from .bag_of_words_machine_translation import BagOfWordsMachineTranslationEvaluator

def Evaluator(task_name):
    """
    A factory class that chooses the evaluator at runtime based on the task name
    """
    task_map = {
        "word_translation": WordTranslationEvaluator,
        "word_translation_with_context": WordTranslationEvaluator,
        "translation_conditioned_language_modeling": TranslationConditionedLanguageModelingEvaluator,
        "bag_of_words_machine_translation": BagOfWordsMachineTranslationEvaluator
    }

    if task_name not in task_map:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(task_map.keys())}"
        )
    
    return task_map[task_name]()
