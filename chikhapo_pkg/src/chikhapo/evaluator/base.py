from abc import abstractmethod
import json
import os
import re
import statistics
from fuzzywuzzy import fuzz
from nltk.corpus import wordnet as wn
from nltk import data, download

from chikhapo import Loader
from chikhapo.utils.constants import min_similarity_score

class BaseEvaluator:
    """
    This class contains the core functionality of all Evaluators, namely the following:
    * reading and performing verification of user input
    * scoring a language based on a series of word scores
    * many helper functions that determine if a translation is correct relative to a 
        list of correct translations
    """
    def __init__(self):
        self.loader = Loader()
        self.DIRECTION = None
        self.word_scores = {}
        self.lang_score = -1
        self.src_lang = None
        self.tgt_lang = None
        self.data = []
        self.verify_wordnet_is_installed()
    
    def verify_wordnet_is_installed(self):
        home_nltk = os.path.expanduser("~/nltk_data")
        if home_nltk not in data.path:
            data.path.append(home_nltk)

        try:
            wn.ensure_loaded()
        except LookupError:
            download("wordnet", download_dir=home_nltk)
            wn.ensure_loaded()

    def get_lang_score(self):
        return self.lang_score
    
    def read_prediction_file(self, file_path):
        if not file_path.endswith("json"):
            raise Exception("The file you provided is not a JSON file. Please input the path to a JSON file")
        with open(file_path, "r") as f:
            model_output = json.load(f)
        if "src_lang" not in model_output.keys():
            raise Exception("The key \"src_lang\" is not specified. Please specify the key to the source language.")
        if not isinstance(model_output["src_lang"], str):
            raise Exception("The source language should be specified as a string.")
        if "tgt_lang" not in model_output.keys():
            raise Exception("The key \"tgt_lang\" is not specified. Please specify the key to the target language.")
        if not isinstance(model_output["tgt_lang"], str):
            raise Exception("The target language should be specified as a string.")
        src_lang = model_output["src_lang"]
        tgt_lang = model_output["tgt_lang"]
        if src_lang == "all" or tgt_lang == "all":
            raise Exception("This function can only evaluate data from one translation. You will have to split your data by language pair and evaluate each split separately.")
        if "data" not in model_output.keys():
            raise Exception("The key \"data\" is not specified. Please specify the key to data.")
        if not isinstance(model_output["data"], list):
            raise Exception("The data you provided does not exist as a list. Please specify the data as a list")
        self.src_lang = model_output["src_lang"]
        self.tgt_lang = model_output["tgt_lang"]
        self.data = model_output["data"]
    
    def get_direction(self):
        if self.tgt_lang=="eng":
            self.DIRECTION = "X_to_eng"
        elif self.src_lang=="eng":
            self.DIRECTION = "eng_to_X"
        else:
            raise Exception("The current implementation of ChiKhaPo's evaluation is English-centric. Model must either translate to or from English (i.e. either self.src_lang==\"eng\" OR self.tgt_lang==\"eng\")")

    def score_language(self):
        word_scores = list(self.word_scores.values())
        if len(word_scores) == 0:
            self.lang_score = 0
        else:
            self.lang_score = round(statistics.mean(word_scores) * 100, 5)
            
    def is_apologetic(self, text):
        text = text.lower()
        return "i'm sorry" in text or "sorry, i cannot" in text

    def no_translation(self, text):
        text = text.lower()
        return "cannot find a translation" in text or "can't find a translation" in text or "no translation" in text or "cannot answer" in text
    
    def is_uncertain(self, text):
        text = text.lower()
        return "i'm not sure" in text or "i have no idea" in text
        
    def de_facto_no_translation(self, text):
        return self.is_apologetic(text) or self.no_translation(text) or self.is_uncertain(text)
    
    def is_exact_match(self, pred, gt_answers):
        for gt_answer in gt_answers:
            if pred == gt_answer:
                return True
        return False

    def is_inflection(self, prediction, gt_answers):
        # ans: ["preguntos"] | prediction: "preguntas"
        if self.de_facto_no_translation(prediction):
            return False
        for ans in gt_answers:
            similarity_score = fuzz.ratio(prediction, ans)
            if similarity_score >= min_similarity_score:
                return True
        return False

    def find_whole_word(self, w):
        return re.compile(r'\b({0})\b'.format(re.escape(w)), flags=re.IGNORECASE).search

    def is_substring(self, prediction, gt_answers):
        # ans: ['good mornings'] | prediction: 'good morning' <-- pure substring
        if self.de_facto_no_translation(prediction):
            return False
        for ans in gt_answers:
            if self.find_whole_word(ans)(prediction):
                return True
        return False

    def is_inflection_within_substring(self, prediction, gt_answers):
        # ans: ['tooths'] | prediction: 'the answer is tooth' <-- an inflection within a string
        if self.de_facto_no_translation(prediction):
            return False
        words_in_prediction = prediction.split()
        for ans in gt_answers:
            for word in words_in_prediction:
                similarity_score = fuzz.ratio(word, ans)
                if similarity_score >= min_similarity_score:
                    return True
        return False

    def preprocess_lemma_names(self, lemmas):
        return [lemma.name() for lemma in lemmas]

    def lemmatize_terms(self, list_of_terms):
        lemma_names = set()
        for term in list_of_terms:
            synsets_of_term = wn.synsets(term)
            for synset_of_term in synsets_of_term:
                lemmas_of_term = synset_of_term.lemmas()
                lemma_names.update(self.preprocess_lemma_names(lemmas_of_term))
        return lemma_names

    def is_synonym(self, prediction, gt_answers):
        # ans: ['tooth'] | prediction: "the answer is incisor" <- a 'synonym' within a string
        # ans: ["dog"] | prediction: "canine"
        if self.DIRECTION != "X_to_eng":
            return False
        if self.de_facto_no_translation(prediction):
            return False
        if len(prediction.split()) > 1:
            list_of_predictions = [prediction] + prediction.split()
        else:
            list_of_predictions = [prediction]
        lemma_names_of_pred = self.lemmatize_terms(list_of_predictions)
        lemma_names_of_gt = self.lemmatize_terms(gt_answers)
        if lemma_names_of_pred & lemma_names_of_gt:
            return True
        return False
    
    @abstractmethod
    def validate_data(self):
        pass

    @abstractmethod
    def clear_intermediary_data(self):
        pass

    @abstractmethod
    def score_each_word(self):
        pass

    @abstractmethod
    def evaluate(self, file_path):
        pass
    