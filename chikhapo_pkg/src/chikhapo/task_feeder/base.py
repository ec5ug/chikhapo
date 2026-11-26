from abc import abstractmethod
import random
from chikhapo import Loader

class BaseTaskFeeder:
    """
    Parent task feeder. Provides functionality of
    * shuffling data
    * retrieving language pairs for which data exists (in a given task)
    * retrieving data for a specific language pair (in a given task)
    * retrieving prompts (in a given task)
    """
    def __init__(self):
        self.loader = Loader()
    
    def get_random_sample(self, d, sample_size=300):
        if len(d) <= sample_size:
            return d
        sampled_keys = random.Random(42).sample(sorted(d), k=sample_size)
        return {k: d[k] for k in sampled_keys}

    @abstractmethod
    def get_lang_pairs(self, DIRECTION=None):
        pass

    @abstractmethod
    def get_data_for_lang_pair(self, lang_pair, lite=True):
        pass

    @abstractmethod
    def get_prompts_for_lang_pair(self, lang_pair, lite=True):
        pass
    