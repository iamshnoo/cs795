import json
from comet import download_model, load_from_checkpoint

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

dict_template = {
    "zh" : "chinese_simplified",
    "hi" : "hindi",
    "fr" : "french",
    "de" : "german"
}

# Paths
lang = "zh"  # ["zh", "hi", "fr", "de"]
lang_path = f"/scratch/amukher6/cs795/sft_data/{dict_template[lang]}.jsonl"
en_path = f"/scratch/amukher6/cs795/sft_data/english.jsonl"

lang_data = load_jsonl(lang_path)
en_data = load_jsonl(en_path)

assert len(lang_data) == len(en_data), "Mismatch between lang and en dataset sizes!"

qe_data = []
for lang_item, en_item in zip(lang_data, en_data):
    lang_text = lang_item["instruction"]
    en_text = en_item["instruction"]

    qe_data.append({
        "src": en_text,       # Source sentence (English)
        "mt": lang_text         # Machine translation (lang)
    })

model_path = download_model(
    "Unbabel/wmt22-cometkiwi-da",
    saving_directory="/scratch/amukher6/cache/huggingface/comet",
)
model = load_from_checkpoint(model_path)
model_output = model.predict(qe_data, batch_size=8, gpus=1)
print(model_output["system_score"])
