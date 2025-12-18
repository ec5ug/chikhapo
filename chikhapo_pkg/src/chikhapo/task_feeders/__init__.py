from .word_translation import WordTranslationFeeder
from .word_translation_with_context import WordTranslationWithContextFeeder
from .fill_in_the_blank import FillInTheBlankFeeder
from .bag_of_words_machine_translation import BagOfWordsMachineTranslationFeeder

def TaskFeeder(task_name):
    task_map = {
        "word_translation": WordTranslationFeeder,
        "word_translation_with_context": WordTranslationWithContextFeeder,
        "fill_in_the_blank": FillInTheBlankFeeder,
        "bag_of_words_machine_translation": BagOfWordsMachineTranslationFeeder
    }
    
    if task_name not in task_map:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(task_map.keys())}"
        )
    
    return task_map[task_name]()