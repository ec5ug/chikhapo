from collections import defaultdict
import json
from fuzzywuzzy import fuzz
from pathlib import Path
import re
import statistics

from chikhapo import Loader

import sys
CURRENT_DIR = Path(__file__).resolve().parent
UTILS_DIR = CURRENT_DIR.parents[2] / "utils"
sys.path.append(str(UTILS_DIR))
from parsing import clean_string, lemmatize_terms
from constants import min_similarity_score

class Evaluator:
    def __init__(self):
        self.loader = Loader()
        self.DIRECTION = None
        self.xword_class_pred = {} # used to be self.model_alignments
        self.word_scores = {}
        self.lang_score = -1
    
    def read_prediction_file(self, file_path):
        if not file_path.endswith("json"):
            raise Exception("The file you provided is not a JSON file. Please input the path to a JSON file")
        with open(file_path, "r") as f:
            model_output_file = json.load(f)
        if "src_lang" not in model_output_file.keys():
            raise Exception("The key \"src_lang\" is not specified. Please specify the key to the source language.")
        if not isinstance(model_output_file["src_lang"], str):
            raise Exception("The source language should be specified as a string.")
        if "tgt_lang" not in model_output_file.keys():
            raise Exception("The key \"tgt_lang\" is not specified. Please specify the key to the target language.")
        if not isinstance(model_output_file["tgt_lang"], str):
            raise Exception("The target language should be specified as a string.")
        if "data" not in model_output_file.keys():
            raise Exception("The key \"data\" is not specified. Please specify the key to data.")
        if not isinstance(model_output_file["data"], list):
            raise Exception("The data you provided does not exist as a list. Please specify the data as a list")
        return model_output_file
    
    def convert_list_of_entries_to_dictionary(self, list_of_entries):
        new_dictionary = defaultdict(list)
        for entry in list_of_entries:
            new_dictionary[entry["source_word"]] = entry["target_translations"]
        return new_dictionary

    def score_language(self): # used to be score_each_word_type
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

    def findWholeWord(self, w):
        return re.compile(r'\b({0})\b'.format(re.escape(w)), flags=re.IGNORECASE).search

    def is_substring(self, prediction, gt_answers):
        # ans: ['good mornings'] | prediction: 'good morning' <-- pure substring
        if self.de_facto_no_translation(prediction):
            return False
        for ans in gt_answers:
            if self.findWholeWord(ans)(prediction):
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
        lemma_names_of_pred = lemmatize_terms(list_of_predictions)
        lemma_names_of_gt = lemmatize_terms(gt_answers)
        # print(prediction, lemma_names_of_gt, lemma_names_of_pred)
        if lemma_names_of_pred & lemma_names_of_gt:
            return True
        return False

    def validate_output(self, elem):
        if "word" not in elem.keys():
            raise Exception(f"One of data points you provided {elem} does not have the word to translate specified. Please take another look at the file you want us to translate and make sure the list elements of the data field are formatted correctly.")
        if "prediction" not in elem.keys():
            raise Exception(f"One of data points you provided {elem} does not have the a (parsed) model prediction to evaluate on. Please take another look at the file you want us to translate and make sure the list elements of the data field are formatted correctly.")
    
    def score_each_word(self):
        for word in self.xword_class_pred:
            exact_match = len(self.xword_class_pred[word].get("exact_match", []))
            inflection = len(self.xword_class_pred[word].get("inflection", []))
            substring = len(self.xword_class_pred[word].get("substring", []))
            inflection_within_substring = len(self.xword_class_pred[word].get("inflection_within_substring", []))
            synonym = len(self.xword_class_pred[word].get("synonym", []))
            # inflected_synonym = len(word_type_data[word].get("inflected_synonym", []))
            # correct = exact_match + inflection + substring + inflection_within_substring + synonym + inflected_synonym
            correct = exact_match + inflection + substring + inflection_within_substring + synonym
            
            echo = len(self.xword_class_pred[word].get("echo", []))
            outputted_in_source_language = len(self.xword_class_pred[word].get("outputted_in_source_language", []))
            gibberish = len(self.xword_class_pred[word].get("gibberish", []))
            incorrect = echo + outputted_in_source_language + gibberish
            
            total = correct + incorrect
            if total==0:
                self.word_scores[word] = 0
            else:
                self.word_scores[word] = correct / total

    def evaluate_word_translation(self, file_path):
        model_output_file = self.read_prediction_file(file_path)
        src_lang = model_output_file["src_lang"]
        tgt_lang = model_output_file["tgt_lang"]
        if src_lang == "all" or tgt_lang == "all":
            raise Exception("This function can only evaluate data from one translation. You will have to split your data by language pair and evaluate each split separately.")
        if tgt_lang == "eng":
            self.DIRECTION = "X_to_eng"
        else:
            self.DIRECTION = "eng_to_X"
        data = model_output_file["data"]
        list_of_entries = self.loader.get_omnis_lexicon_subset(f"{src_lang}_{tgt_lang}")
        lexicon = self.convert_list_of_entries_to_dictionary(list_of_entries)
        for output in data:
            self.validate_output(output)
            word_to_translate = clean_string(output["word"])
            if word_to_translate not in lexicon.keys():
                continue
            gt_answers = lexicon[word_to_translate]
            prediction = clean_string(output["prediction"])
            if self.is_exact_match(prediction, gt_answers):
                classification_type = "exact_match"
            elif self.is_inflection(prediction, gt_answers):
                classification_type = "inflection"
            elif self.is_substring(prediction, gt_answers):
                classification_type = "substring"
            elif self.is_inflection_within_substring(prediction, gt_answers):
                classification_type = "inflection_within_substring"
            elif self.is_synonym(prediction, gt_answers):
                classification_type = "synonym"
            elif word_to_translate == prediction:
                classification_type = "echo"
            elif prediction in lexicon.keys():
                classification_type = "outputted_in_source_language"
            else:
                classification_type = "gibberish"

            if self.DIRECTION=="X_to_eng":
                x_words = [word_to_translate]
            elif self.DIRECTION=="eng_to_X":
                x_words = gt_answers # -> X
            
            for x_word in x_words: # allows for synonymy
                if x_word not in self.xword_class_pred:
                    self.xword_class_pred[x_word] = {}
                if classification_type not in self.xword_class_pred[x_word]:
                    self.xword_class_pred[x_word][classification_type] = []
                self.xword_class_pred[x_word][classification_type].append(prediction)
        self.score_each_word()
        self.score_language() # used to be self.score_each_word_type()