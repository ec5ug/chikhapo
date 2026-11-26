from chikhapo.utils.languages import get_language_from_pair
from chikhapo.utils.parsing import clean_string, preprocess_string, convert_list_of_entries_to_dictionary, is_source_sentence
from .word_translation import WordTranslationEvaluator

class WordTranslationWithContextEvaluator(WordTranslationEvaluator):
    """
    The evaluator for Word Translation With Context. Scoring is identical to Word
    Translation (score_each_word, validate_data, and the is_* helpers are inherited).
    The only difference is how a word is looked up in the lexicon: words here come
    from running text, so they may be capitalized (e.g. at the start of a sentence)
    or carry punctuation from the sentence (e.g. "bahasa,").
    """
    def resolve_source_word(self, word, lexicon):
        raw = word.strip()
        for candidate in (raw, raw.lower(), clean_string(raw)):
            if candidate and candidate in lexicon:
                return candidate
        return None

    def validate_data(self):
        super().validate_data()
        for entry in self.data:
            if "source_sentence" not in entry:
                raise Exception(f"A source sentence was not specified in {entry}")

    def is_in_src_language(self, prediction, lexicon):
        for w in prediction.split():
            cleaned_w = clean_string(w)
            if cleaned_w in lexicon and len(cleaned_w) > 3:
                return True
        return False
        
    def score_each_word(self):
        correct_types = ("exact_match", "inflection", "substring",
                         "inflection_within_substring", "synonym")
        incorrect_types = ("echo", "outputted_src_sentence",
                           "outputted_in_source_language", "gibberish")
        for word, classes in self.xword_class_pred.items():
            correct = sum(len(classes.get(t, [])) for t in correct_types)
            incorrect = sum(len(classes.get(t, [])) for t in incorrect_types)
            total = correct + incorrect
            self.word_scores[word] = 0 if total == 0 else correct / total

    def evaluate(self, file_path):
        self.read_prediction_file(file_path)
        self.validate_data()
        lang = get_language_from_pair(f"{self.src_lang}_{self.tgt_lang}")
        list_of_entries = self.loader.get_chikhapo_lexicon_subset(lang, self.get_direction())
        lexicon = convert_list_of_entries_to_dictionary(list_of_entries)
        for output in self.data:
            # word_to_translate = self.resolve_source_word(output["word"], lexicon)
            word_to_translate = output["word"].lower()
            if word_to_translate not in lexicon:
                continue
            gt_answers = [preprocess_string(answer) for answer in lexicon[word_to_translate]]
            prediction = clean_string(output["prediction"])
            words_in_prediction = prediction.split()
            if self.is_exact_match(prediction, gt_answers):
                classification_type = "exact_match"
            elif is_source_sentence(prediction, output["source_sentence"]):
                classification_type = "outputted_src_sentence"
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
            elif self.is_in_src_language(prediction, lexicon):
                classification_type = "outputted_in_source_language"
            else:
                classification_type = "gibberish"

            if self.DIRECTION=="X_to_eng":
                x_words = [word_to_translate]
            elif self.DIRECTION=="eng_to_X":
                # x_words = gt_answers # -> X
                x_words = [clean_string(answer) for answer in lexicon[word_to_translate]]

            for x_word in x_words: # allows for synonymy
                if x_word not in self.xword_class_pred:
                    self.xword_class_pred[x_word] = {}
                if classification_type not in self.xword_class_pred[x_word]:
                    self.xword_class_pred[x_word][classification_type] = []
                self.xword_class_pred[x_word][classification_type].append(prediction)
        self.score_each_word()
        self.score_language()