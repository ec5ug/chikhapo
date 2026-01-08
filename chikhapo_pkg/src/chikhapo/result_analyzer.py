import os
import pandas as pd
import pycountry
import statistics
import warnings
import pprint
from collections import defaultdict

from chikhapo import Evaluator, GlottologReader

class ResultAnalyzer:
    def __init__(self, task_name):
        self.task_name = task_name
        self.evaluator = Evaluator(self.task_name)
        self.results_by_language = {}
        self.results_by_language_family = {}
        self.glottolog_reader = GlottologReader()

    def get_results_by_language(self, result_dir):
        if not os.path.isdir(result_dir):
            raise Exception(f"The path {result_dir} is not a valid directory.")
        if len(os.listdir(result_dir))==0:
            warnings.warn("This directory is empty!")

        for filename in os.listdir(result_dir):
            full_path = os.path.join(result_dir, filename)
            self.evaluator.clear_intermediary_data()
            self.evaluator.evaluate(full_path)
            # print(filename.split(".")[0])
            # print(pprint.pformat(self.evaluator.xword_probs))
            # print(pprint.pformat(self.evaluator.xword_class_pred))
            # print("-" * 100)
            if self.evaluator.src_lang=="eng" and self.evaluator.tgt_lang=="eng":
                raise Exception("The language pair eng-eng is invalid")
            elif self.evaluator.src_lang=="eng":
                lang = self.evaluator.tgt_lang
            elif self.evaluator.tgt_lang=="eng":
                lang = self.evaluator.src_lang
            else:
                raise Exception("ResultAnalyzer can only process language pairs "\
                                "translate to OR from English.")
            if not pycountry.languages.get(alpha_3=lang):
                raise Exception(f"{filename}: There is a language field that is an invalid "\
                                "ISO code.")
            self.results_by_language[lang] = self.evaluator.lang_score
        
        if not len(self.results_by_language):
            warnings.warn("Unfortunately, the directory you provided did not yield any data that could be evaluated. The dictionary associated with results by language is subsequently empty.")
    
    def get_language_score_average(self):
        if not len(self.results_by_language):
            raise Exception("The dictionary results_by_language is completely empty. Consequently, the language score cannot be calculated.")
    
        scores = self.results_by_language.values()
        avg = statistics.mean(scores)
        return avg
    
    def get_language_score_standard_deviation(self):
        if not len(self.results_by_language):
            raise Exception("The dictionary results_by_language is completely empty. Consequently, the language score cannot be calculated.")
        scores = self.results_by_language.values()
        std_dev = statistics.stdev(scores)
        return std_dev

    def get_results_by_language_family(self):
        if not self.results_by_language:
            raise Exception(f"Before you can attain results by language family, you must "\
                            "have results for individual languages. You must call "\
                            ".get_lang_results(result_dir) with a valid results directory prior "\
                            "to calling .get_language_family_results()")
        language_to_family = self.glottolog_reader.get_language_to_family_dict()
        for lang, score in self.results_by_language.items():
            fam = language_to_family[lang]
            # print(lang, fam)
            if fam not in self.results_by_language_family:
                self.results_by_language_family[fam] = {
                    "scores": [],
                    "avg": -1,
                    "std_dev": -1
                }
            self.results_by_language_family[fam]["scores"].append(score)
        for fam in self.results_by_language_family:
            scores = self.results_by_language_family[fam]["scores"]
            self.results_by_language_family[fam]["avg"] = statistics.mean(scores)
            if len(scores) > 1:
                self.results_by_language_family[fam]["std_dev"] = statistics.stdev(scores)
            else:
                warnings.warn(f"Only one language fell into the language family {fam}. You need at least two to calculate the standard deviation. Setting the standard deviation of this langugae family to -1.")
                self.results_by_language_family[fam]["std_dev"] = -1
            # print(scores, statistics.mean(scores), statistics.stdev(scores))
    