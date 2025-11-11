import json, os, yaml
from huggingface_hub import HfApi
from chikhapo.utils.languages import get_language_pair

# valid_langs = set()
# for DIRECTION in ["X_to_eng"]:
#     for filename in os.listdir(f"../omnis/{DIRECTION}"):
#         lang_pair, _ = os.path.splitext(filename)
#         if lang_pair == "eng-eng":
#             continue
#         lang_x = lang_pair.split("-")[0]

#         x_to_eng_path = f"../omnis/X_to_eng/{lang_x}-eng.json"
#         eng_to_x_path = f"../omnis/eng_to_X/eng-{lang_x}.json"
        
#         # Check both files exist
#         if not os.path.exists(x_to_eng_path) or not os.path.exists(eng_to_x_path):
#             continue

#         with open(x_to_eng_path, "r") as f:
#             x_to_eng_lex = json.load(f)

#         with open(eng_to_x_path, "r") as f:
#             eng_to_x_lex = json.load(f)

#         # Check both have sufficient entries
#         if len(x_to_eng_lex) >= 100:
#             valid_langs.add(lang_x)

REPO_NAME = "ec5ug/chikhapo"
DATA_DIR = "../hf_data/data"
os.makedirs(DATA_DIR, exist_ok=True)

# configs = []

# for DIRECTION in ["X_to_eng", "eng_to_X"]:
#     all_records = [] # for creating all_eng and eng_all
#     counter = 0
#     for lang in valid_langs:
#         counter += 1
#         if DIRECTION == "X_to_eng":
#             filename = f"{lang}-eng.json"
#         else:
#             filename = f"eng-{lang}.json"
#         if DIRECTION == "X_to_eng":
#             src_lang, tgt_lang = lang, "eng"
#         else:
#             src_lang, tgt_lang = "eng", lang
#         config_name = f"{src_lang}_{tgt_lang}"

#         with open(f"../omnis/{DIRECTION}/{filename}", "r") as f:
#             lexicon = json.load(f)
        
#         # Reformatting Lexicons
#         # Please be VERY WARY when running this command. It is very expensive
#         #   both in terms of time and memory.
#         output_path = os.path.join(DATA_DIR, f"{config_name}.jsonl")
#         with open(output_path, "w", encoding="utf-8") as out:
#             for w, t in lexicon.items():
#                 if not t:
#                     continue
#                 record = {
#                     "source_word": w,
#                     "target_translations": list(t.keys()),
#                     "src_lang": src_lang,
#                     "tgt_lang": tgt_lang,
#                 }
#                 out.write(json.dumps(record, ensure_ascii=False) + "\n")
        
#         for w, t in lexicon.items():
#             if not t:
#                 continue
#             record = {
#                 "source_word": w,
#                 "target_translations": list(t.keys()),
#                 "src_lang": src_lang,
#                 "tgt_lang": tgt_lang,
#             }
#             all_records.append(record)

#         configs.append({
#             "config_name": config_name,
#             "data_files": [f"data/{config_name}.jsonl"]
#         })

#     print(f"{DIRECTION}: {counter} language pairs")
#     if DIRECTION=="X_to_eng":
#         config_name = "all_eng"
#     else:
#         config_name = "eng_all"
    
#     # for dumping all_eng or eng_all
#     output_path = os.path.join(DATA_DIR, f"{config_name}.jsonl")
#     with open(output_path, "w", encoding="utf-8") as out:
#         for record in all_records:
#             out.write(json.dumps(record, ensure_ascii=False) + "\n")
#     configs.append({
#         "config_name": config_name,
#         "data_files": [f"data/{config_name}.jsonl"]
#     })

# # creating the YAML based on language and splits
# yaml_path = os.path.join("../dataset_config.yaml")

# with open(yaml_path, "w", encoding="utf-8") as f:
#     f.write("\n")
#     f.write("---\n")
#     f.write("license: mit\n")
#     f.write("pretty_name: ChiKhaPo\n")
#     f.write("language:\n")
#     for lang in valid_langs:
#         f.write(f"  - {lang}\n")
#     f.write("configs:\n")
#     for config in configs:
#         f.write(f"  - config_name: {config['config_name']}\n")
#         f.write(f"    data_files: {config['data_files'][0]}\n")
#     f.write("---\n")
#     f.write("\n")

# Upload folder
api = HfApi()
api.upload_large_folder(
    folder_path="../hf_data/",
    repo_id=REPO_NAME,
    repo_type="dataset",
)

"""
# Introduction
Our benchmark is described in [ChiKhaPo: A Large-Scale Multilingual Benchmark for Evaluating Lexical Comprehension and Generation in Large Language Models](https://www.arxiv.org/abs/2510.16928).
ChiKhaPo contains 4 word-level tasks, with two directions each (comprehension and generation), intended to benchmark generative models for lexical competence. The dataset itself contains the the lexicons that our tasks rely on. See our [GitHub](https://github.com/ec5ug/chikhapo) or [pip package]() for instructions on running our benchmark

# Dataset
**Subset Names**: The subset names in this dataset correspond to the language pairs they represent. The source and target language are represented as ISO-3 codes and separated by an `_`. For example, the subset name to retrieve the Spanish-English dataset is `spa_eng`.

You can retrieve a list of all language pairs included using
```
from datasets import get_dataset_config_names
config_names = get_dataset_config_names("ec5ug/chikhapo")
```

This dataset covers 2750 languages. Refer to [our paper](https://www.arxiv.org/abs/2510.16928) for further details on statistics.

**Entries**: The dataset consists of a list of dictionaries, each containing the keys `source_word`, `target_translations`, `src_lang`, and `tgt_lang`. An example entry is shown below:

```
{
  "source_word": "morot",
  "target_translations": ["person", "man"],
  "src_lang": "aot", 
  "tgt_lang": "eng"
}
```

**Retrieving all datasets**: Use the subset name `all_eng` to retrieve all datasets that translate from any language into English. Conversely, use `eng_all` to access datasets that translate from English into all other languages.

# Cite
If you use this data or code, please cite
```
@article{chang2025chikhapo,
  title={ChiKhaPo: A Large-Scale Multilingual Benchmark for Evaluating Lexical Comprehension and Generation in Large Language Models},
  author={Chang, Emily and Bafna, Niyati},
  journal={arXiv preprint arXiv:2510.16928},
  year={2025}
}
```
"""