from collections import defaultdict
import unicodedata
import re
import string
from fuzzywuzzy import fuzz

def is_punctuation(char):
    return unicodedata.category(char).startswith('P')

def strip_punctuation(word):
    chars = list(word)
    while chars and is_punctuation(chars[0]):
        chars.pop(0)
    while chars and is_punctuation(chars[-1]):
        chars.pop()
    return ''.join(chars)

def clean_string(text):
    text = text.lower()
    text = text.strip()
    text = strip_punctuation(text)
    text = text.strip()
    return text

def convert_list_of_entries_to_dictionary(list_of_entries):
    new_dictionary = defaultdict(list)
    for entry in list_of_entries:
        new_dictionary[entry["source_word"]] = entry["target_translations"]
    return new_dictionary

def find_paired_quotes(text):
    results = []
    left_index = None
    quote_char = None
    
    i = 0
    while i < len(text):
        char = text[i]
        if char in ("'", '"'):
            if left_index is None:
                # opening quote
                if i == 0 or text[i-1] == " ":
                    left_index = i
                    quote_char = char
            else:
                if char == quote_char:
                    # handle doubled quotes: skip treating as close if next char is the same
                    if i+1 < len(text) and text[i+1] == char:
                        i += 1   # skip this one, treat as doubled
                        continue
                    # closing quote conditions
                    if (i == len(text)-1) or (text[i+1] == " " or text[i+1] in string.punctuation):
                        results.append(text[left_index:i+1])
                        left_index = None
                        quote_char = None
        i += 1
    return results

def preprocess_string(word, text_to_be_removed=None):
    if text_to_be_removed:
        word = word.replace(text_to_be_removed, "")
    word = word.replace('/', ' ')
    word = re.sub(r'<.+?>', '', word)
    word = word.translate(str.maketrans('', '', string.punctuation))
    word = word.lower()
    word = word.strip()
    word = re.sub(r'\s+', ' ', word)
    return word

def contains_exactly_one_letter(word):
    contains_exactly_one_letter = False
    for i in range(len(word)):
        char = word[i]
        if char.isalpha():
            contains_exactly_one_letter = True
            word = word[:i] + word[i+1:]
            break
    for i in range(len(word)):
        char = word[i]
        if char.isalpha():
            return False
    return contains_exactly_one_letter

def is_source_sentence(prediction, src_sentence):
    prediction = clean_string(prediction)
    src_sentence = clean_string(src_sentence)
    similarity_score = fuzz.ratio(prediction, src_sentence)
    return similarity_score >= 91