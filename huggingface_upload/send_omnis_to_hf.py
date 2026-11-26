# from huggingface_hub import login, logout
# logout()
# login()
# login(token="REMOVED")

import json, os, yaml
from huggingface_hub import HfApi
from utils.languages import read_word_translation_valid_langs

REPO_NAME = "ec5ug/chikhapo"
DATA_DIR = "../hf_data/data"
yaml_path = "../dataset_config.yaml"
os.makedirs(DATA_DIR, exist_ok=True)

valid_langs = read_word_translation_valid_langs()

all_records = []
configs = []

for lang in valid_langs:
    if lang == "eng":
        continue
    config_name = lang
    for DIRECTION in ["X_to_eng", "eng_to_X"]:
        if DIRECTION=="X_to_eng":
            src_lang, tgt_lang = lang, "eng"
        else:
            src_lang, tgt_lang = "eng", lang
        filename = f"{src_lang}-{tgt_lang}.json"
        with open(f"../omnis/{DIRECTION}/{filename}", "r", encoding="utf-8") as f:
            lexicon = json.load(f)

        expected_line_count = sum(1 for t in lexicon.values() if t)
        if os.path.exists(os.path.join(DATA_DIR, f"{src_lang}-{tgt_lang}.jsonl")):
            with open(os.path.join(DATA_DIR, f"{src_lang}-{tgt_lang}.jsonl"), encoding="utf-8") as file:
                if sum(1 for _ in file) == expected_line_count:
                    continue
        
        # Reformatting Lexicons
        # Please be VERY WARY when running this command. It is very expensive
        #   both in terms of time and memory.
        out_path = os.path.join(DATA_DIR, f"{src_lang}-{tgt_lang}.jsonl")
        lite_idx = {int(j * len(lexicon) / 300) for j in range(300)}
        with open(out_path, "w", encoding="utf-8") as out:
            for i, (w, t) in enumerate(lexicon.items()):
                if not t:
                    continue
                record = {
                    "source_word": w,
                    "target_translations": list(t.keys()),
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "lite": len(lexicon) <= 300 or i in lite_idx,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    configs.append({
        "config_name": config_name,
        "data_files": [{
            "split": "X_to_eng",
            "path": f"data/{lang}-eng.jsonl"
        }, {
            "split": "eng_to_X",
            "path": f"data/eng-{lang}.jsonl"
        }]
    })
for all_name, pair_name in [
    ("all-eng.jsonl", lambda l: f"{l}-eng.jsonl"),
    ("eng-all.jsonl", lambda l: f"eng-{l}.jsonl"),
]:
    with open(os.path.join(DATA_DIR, all_name), "w", encoding="utf-8") as out:
        for lang in valid_langs:
            if lang == "eng":
                continue
            with open(os.path.join(DATA_DIR, pair_name(lang)), encoding="utf-8") as src:
                for line in src:
                    out.write(line)
configs.append({
    "config_name": "all",
    "data_files": [{
        "split": "X_to_eng",
        "path": f"data/all-eng.jsonl"
    }, {
        "split": "eng_to_X",
        "path": f"data/eng-all.jsonl"
    }]
})

# creating the YAML based on language and splits
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write("---\n")
    f.write("license: mit\n")
    f.write("pretty_name: ChiKhaPo\n")
    f.write("tags:\n")
    f.write("  - PanLex\n")
    f.write("  - lexicon\n")
    f.write("  - multilingual\n")
    f.write("language:\n")
    for lang in valid_langs:
        f.write(f"  - {lang}\n")
    f.write("configs:\n")
    for config in configs:
        f.write(f"  - config_name: {config['config_name']}\n")
        f.write("    data_files:\n")
        for split in config["data_files"]:
            f.write(f"    - split: {split['split']}\n")
            f.write(f"      path: {split['path']}\n")
    f.write("---\n\n")
    
# Upload folder
api = HfApi()
api.delete_folder(path_in_repo="data", repo_id=REPO_NAME, repo_type="dataset")
api.upload_large_folder(
    folder_path="../hf_data/",
    repo_id=REPO_NAME,
    repo_type="dataset",
)
