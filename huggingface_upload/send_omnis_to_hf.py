import json, os, yaml
from huggingface_hub import HfApi

REPO_NAME = "ec5ug/chikhapo"
DATA_DIR = "hf_data/data"
os.makedirs(DATA_DIR, exist_ok=True)

configs = []
langs = set()

for DIRECTION in ["X_to_eng", "eng_to_X"]:
    all_records = [] # for creating all_eng and eng_all
    for filename in os.listdir(f"omnis/{DIRECTION}"):
        lang_pair, _ = os.path.splitext(filename)
        src_lang, tgt_lang = lang_pair.split("-")
        config_name = f"{src_lang}_{tgt_lang}"

        with open(f"omnis/{DIRECTION}/{filename}", "r") as f:
            lexicon = json.load(f)

        if len(lexicon) < 99:
            continue
        
        # Reformatting Lexicons
        # Please be VERY WARY when running this command. It is very expensive
        #   both in terms of time and memory.
        output_path = os.path.join(DATA_DIR, f"{config_name}.jsonl")
        with open(output_path, "w", encoding="utf-8") as out:
            for w, t in lexicon.items():
                if not t:
                    continue
                record = {
                    "source_word": w,
                    "target_translations": list(t.keys()),
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        for w, t in lexicon.items():
            if not t:
                continue
            record = {
                "source_word": w,
                "target_translations": list(t.keys()),
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
            }
            all_records.append(record)

        configs.append({
            "config_name": config_name,
            "data_files": [f"data/{config_name}.jsonl"]
        })
        langs.add(src_lang)

    if DIRECTION=="X_to_eng":
        config_name = "all_eng"
    else:
        all_filename = "eng_all"
    
    # for dumping all_eng or eng_all
    output_path = os.path.join(DATA_DIR, f"{config_name}.jsonl")
    with open(output_path, "w", encoding="utf-8") as out:
        for record in all_records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    configs.append({
        "config_name": config_name,
        "data_files": [f"data/{config_name}.jsonl"]
    })

# creating the YAML based on language and splits
langs = list(langs)
print(len(langs))
yaml_path = os.path.join("dataset_config.yaml")

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write("\n")
    f.write("---\n")
    f.write("license: mit\n")
    f.write("pretty_name: ChiKhaPo\n")
    f.write("language:\n")
    for lang in langs:
        f.write(f"  - {lang}\n")
    f.write("configs:\n")
    for config in configs:
        f.write(f"  - config_name: {config['config_name']}\n")
        f.write(f"    data_files: {config['data_files'][0]}\n")
    f.write("---\n")
    f.write("\n")

# Upload folder
api = HfApi()
api.upload_large_folder(
    folder_path="hf_data/",
    repo_id=REPO_NAME,
    repo_type="dataset",
)
