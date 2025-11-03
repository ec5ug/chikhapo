from abc import abstractmethod
import random
from chikhapo import Loader

class BaseTaskFeeder:
    """
    Base class for all task feeders
    """
    def __init__(self):
        self.loader = Loader()
    
    def get_random_sample(self, list_to_sample, sample_size=300):
        if len(list_to_sample) <= sample_size:
            return list_to_sample
        random.seed(42)
        random.shuffle(list_to_sample)
        return list_to_sample[:sample_size]

    @abstractmethod
    def get_data_for_lang_pair(self, lang_pair, lite=True):
        pass

    @abstractmethod
    def get_prompts_for_lang_pair(self, model_name, lang_pair, lite=True):
        pass