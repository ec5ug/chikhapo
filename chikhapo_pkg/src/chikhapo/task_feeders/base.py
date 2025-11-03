from abc import ABC, abstractmethod
from pathlib import Path
from chikhapo import Loader
import sys

class BaseTaskFeeder:
    """Base class for all task feeders"""
    def __init__(self):
        self.loader = Loader()
        
    @abstractmethod
    def get_data_for_lang_pair(self, lang_pair):
        pass

    @abstractmethod
    def get_prompts_for_lang_pair(self, model_name, lang_pair):
        pass