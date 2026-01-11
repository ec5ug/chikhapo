from collections import defaultdict
import pprint
import statistics

from chikhapo.utils.parsing import clean_string, convert_list_of_entries_to_dictionary
from .base_alignments import BaseAlignmentsEvaluator

class TranslationConditionedLanguageModelingEvaluator(BaseAlignmentsEvaluator):
    """
    As a child of the class Base Alignments Evaluator, the evaluator Translation Conditioned 
    Language Modeling leverages lexicons and statistical alignments to calculate the average 
    probability of a model predicting a word in a sentence 
    """
    def __init__(self):
        super().__init__()
        self.xword_probs = defaultdict(list)
    
    def clear_intermediary_data(self):
        self.xword_probs = defaultdict(list)

    def validate_data(self):
        for entry in self.data:
            entry_str = pprint.pformat(entry)
            if "src_sentence" not in entry:
                raise Exception(f"The key \"src_sentence\" is not found in {entry_str}")
            if not isinstance(entry["src_sentence"], str):
                raise Exception(f"{entry_str}: The source sentence in \"src_sentence\" should be stored as a string, not {type(entry['src_sentence'])}")
            if "tgt_sentence_gt" not in entry:
                raise Exception(f"The key \"tgt_sentence_gt\" is not found in {entry_str}")
            if not isinstance(entry["tgt_sentence_gt"], str):
                raise Exception(f"{entry_str}: The ground-truth target sentence in \"tgt_sentence_gt\" should be stored as a string, not {type(entry['tgt_sentence_gt'])}")
            if "next_word_to_predict" not in entry:
                raise Exception(f"The key \"next_word_to_predict\" is not found in {entry_str}")
            if not isinstance(entry['next_word_to_predict'], str):
                raise Exception(f"{entry_str}: The next word to predict in \"next_word_to_predict\" should be stored as a string, not {type(entry['next_word_to_predict'])}")
            if "probability" not in entry:
                raise Exception(f"The key \"probability\" is not found in {entry_str}")
            if not isinstance(entry["probability"], (float, int)):
                raise Exception(f"{entry_str}: The value of the key \"probability\" should be represented as a float.")
            if entry["probability"] < 0 or entry["probability"] > 1:
                raise Exception(f"The probability measure {entry['probability']} should be in the interval of 0 and 1.")

    def evaluate_X_to_eng(self):
        list_of_entries = self.loader.get_omnis_lexicon_subset(f"{self.tgt_lang}_{self.src_lang}") # flip translation_order
        lexicon = convert_list_of_entries_to_dictionary(list_of_entries)
        alignments = self.get_statistical_alignments(reverse=True)
        for i, entry in enumerate(self.data):
            # cleaning up src_sentence
            src_words = entry["src_sentence"].split()
            src_words = [clean_string(w) for w in src_words]
            eng_word = entry["next_word_to_predict"]
            cleaned_eng_word = clean_string(eng_word)
            prob = entry["probability"]
            xwords = []
            if cleaned_eng_word in lexicon:
                potential_xwords = lexicon[cleaned_eng_word]
                for potential_xword in potential_xwords:
                    cleaned_xword = clean_string(potential_xword)
                    if self.is_exact_match(cleaned_xword, src_words) or self.is_substring(cleaned_xword, src_words) or self.is_inflection(cleaned_xword, src_words) or self.is_inflection_within_substring(cleaned_xword, src_words):
                        xwords.append(cleaned_xword)
            elif eng_word in alignments[i]:
                xwords_scores = alignments[i][eng_word]
                max_score = max(xwords_scores.values())
                xwords = [xword for xword, score in xwords_scores.items() if score==max_score]
            cleaned_xwords = [clean_string(xword) for xword in xwords]
            for xword in cleaned_xwords:
                self.xword_probs[xword].append(prob)
    
    def evaluate_eng_to_X(self):
        for entry in self.data:
            next_word_to_predict = clean_string(entry["next_word_to_predict"])
            prob = entry["probability"]
            self.xword_probs[next_word_to_predict].append(prob)
    
    def score_each_word(self):
        for word, probs in self.xword_probs.items():
            self.word_scores[word] = statistics.mean(probs)

    def evaluate(self, file_path):
        self.read_prediction_file(file_path)
        self.validate_data()
        self.get_direction()
        # manipulate self.word_probs -> self.word_scores
        if self.DIRECTION=="X_to_eng":
            self.evaluate_X_to_eng()
        else: # self.DIRECTION=="eng_to_X"
            self.evaluate_eng_to_X()    
        self.score_each_word()
        self.score_language()
