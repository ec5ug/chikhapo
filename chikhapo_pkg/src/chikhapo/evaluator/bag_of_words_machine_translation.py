import pprint

from chikhapo.utils.parsing import clean_string, convert_list_of_entries_to_dictionary
from .base_alignments import BaseAlignmentsEvaluator

class BagOfWordsMachineTranslationEvaluator(BaseAlignmentsEvaluator):
    """
    The evaluator of the task Bag of Words Machine Translation finds the total number of 
    accurate (source) words. The user provides a file containing source sentence, target 
    ground-truth translations, as well as model translations that are to be evaluated. 
    Because the class is a child of BaseAlignmentsEvaluator, the evaluation then leverages 
    lexicons cand statistical alignments to check the accuracy.
    """
    
    def __init__(self):
        super().__init__()
        self.xword_class_pred = {}

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
            if "tgt_sentence_pred" not in entry:
                raise Exception(f"The key \"tgt_sentence_pred\" is not found in {entry_str}")
            if not isinstance(entry["tgt_sentence_pred"], str):
                raise Exception(f"{entry_str}: The predicted target sentence in \"tgt_sentence_pred\" should be stored as a string, not {type(entry['tgt_sentence_pred'])}")

    def evaluate_X_to_eng(self):
        list_of_entries = self.loader.get_omnis_lexicon_subset(f"{self.src_lang}_{self.tgt_lang}") # flip translation_order
        lexicon = convert_list_of_entries_to_dictionary(list_of_entries)
        alignments = self.get_statistical_alignments(reverse=False)
        for i, entry in enumerate(self.data):
            src_words = entry["src_sentence"].split() # in X
            cleaned_src_words = [clean_string(word) for word in src_words]
            pred_words = entry["tgt_sentence_pred"].split() # in eng
            cleaned_pred_words = [clean_string(word) for word in pred_words]
            for j, cleaned_src_word in enumerate(cleaned_src_words):
                gt_words = []
                src_word = src_words[j]
                if cleaned_src_word in lexicon:
                    gt_words = lexicon[cleaned_src_word]
                elif src_word in alignments[i]:
                    xwords_scores = alignments[i][src_word]
                    max_score = max(xwords_scores.values())
                    gt_words = [xword for xword, score in xwords_scores.items() if score==max_score]
                if len(gt_words) == 0:
                    continue
                cleaned_gt_words = [clean_string(word) for word in gt_words] # in eng
                cleaned_pred_word = ""
                classification_type = "incorrect"
                if classification_type=="incorrect":
                    for cleaned_pred_word in cleaned_pred_words:
                        if self.is_exact_match(cleaned_pred_word, cleaned_gt_words):
                            classification_type = "exact_match"
                            break
                if classification_type=="incorrect":
                    for cleaned_pred_word in cleaned_pred_words:
                        if self.is_inflection(cleaned_pred_word, cleaned_gt_words):
                            classification_type = "inflection"
                            break
                if classification_type=="incorrect":
                    for cleaned_pred_word in cleaned_pred_words:
                        if self.is_synonym(cleaned_pred_word, cleaned_gt_words):
                            classification_type = "synonym"
                            break
                if cleaned_src_word not in self.xword_class_pred:
                    self.xword_class_pred[cleaned_src_word] = {}
                if classification_type not in self.xword_class_pred[cleaned_src_word]:
                    self.xword_class_pred[cleaned_src_word][classification_type] = []
                self.xword_class_pred[cleaned_src_word][classification_type].append(cleaned_pred_word)
    
    def evaluate_eng_to_X(self):
        for entry in self.data:
            gt_words = entry["tgt_sentence_gt"].split() # in X
            cleaned_gt_words = [clean_string(word) for word in gt_words]
            pred_words = entry["tgt_sentence_pred"].split() # in X
            cleaned_pred_words = [clean_string(word) for word in pred_words]
            for cleaned_gt_word in cleaned_gt_words:
                cleaned_pred_word = ""
                classification_type = "incorrect"

                if classification_type=="incorrect":
                    for cleaned_pred_word in cleaned_pred_words:
                        if self.is_exact_match(cleaned_pred_word, [cleaned_gt_word]):
                            classification_type = "exact_match"
                            break
                if classification_type=="incorrect":
                    for cleaned_pred_word in cleaned_pred_words:
                        if self.is_inflection(cleaned_pred_word, [cleaned_gt_word]):
                            classification_type = "inflection"
                            break
                if cleaned_gt_word not in self.xword_class_pred:
                    self.xword_class_pred[cleaned_gt_word] = {}
                if classification_type not in self.xword_class_pred[cleaned_gt_word]:
                    self.xword_class_pred[cleaned_gt_word][classification_type] = []
                self.xword_class_pred[cleaned_gt_word][classification_type].append(cleaned_pred_word)

    def clear_intermediary_data(self):
        self.xword_class_pred = {}

    def score_each_word(self):
        for word in self.xword_class_pred:
            exact_match = len(self.xword_class_pred[word].get("exact_match", []))
            inflection = len(self.xword_class_pred[word].get("inflection", []))
            substring = len(self.xword_class_pred[word].get("substring", []))
            synonym = len(self.xword_class_pred[word].get("synonym", []))
            correct = exact_match + inflection + substring + synonym
            incorrect = len(self.xword_class_pred[word].get("incorrect", []))
            
            total = correct + incorrect
            if total==0:
                self.word_scores[word] = 0
            else:
                self.word_scores[word] = correct / total

    def evaluate(self, file_path):
        self.read_prediction_file(file_path)
        self.validate_data()
        self.get_direction()
        if self.DIRECTION=="X_to_eng":
            self.evaluate_X_to_eng()
        else: # self.DIRECTION=="eng_to_X"
            self.evaluate_eng_to_X()
        self.score_each_word()
        self.score_language()
        