from pathlib import Path
import json

omnis_docs_path = Path(__file__).resolve().parents[1] / "metadata" / "omnis_docs.json"
def read_word_translation_valid_langs(): # used to be read_omnis_valid_langs
    with open(omnis_docs_path, "r") as f:
        omnis_docs = json.load(f)
    omnis_langs = list(set(omnis_docs["valid_langs"]))
    return omnis_langs