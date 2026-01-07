import unicodedata
from collections import defaultdict

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
