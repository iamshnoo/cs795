import os
import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import torch
import re
from tqdm import tqdm
from itertools import product


# ============================================================
# ARGPARSE — mode, batch size, think tokens
# ============================================================
parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, required=True, choices=["baseline", "sft", "grpo"])
parser.add_argument("--num_think_tokens", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=4)

args = parser.parse_args()

mode = args.mode
num_think_tokens = args.num_think_tokens
batch_size = args.batch_size


# ============================================================
# Model path patterns
# ============================================================

MODEL_PATHS = {
    "baseline": "Qwen/Qwen3-4B",
    "sft":      "/scratch/amukher6/cs795/models/{lang}_model2_sft",
    "grpo":     "/scratch/amukher6/cs795/models/{lang}_model3_sft_grpo",
}


# ============================================================
# Determine model version name
# ============================================================

model_version = {
    "baseline": "model1",
    "sft":      "model2",
    "grpo":     "model3",
}[mode]

think_folder = f"t{num_think_tokens}"

# ============================================================
# Dataset locations
# ============================================================
hard_dict = {
    "hard":   "/scratch/amukher6/cs795/test_data/hard",
    "medium": "/scratch/amukher6/cs795/test_data/medium",
}

GLOBAL_SYSTEM = (
    "You are given a multiple-choice question. "
    "Read the context and the question carefully, then answer with only the correct option letter: A, B, or C."
)



# ============================================================
# Loading helpers
# ============================================================

def load_single_test_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                ex = {
                    "example_id": row.get("example_id"),
                    "question_index": row.get("question_index"),
                    "question_polarity": row.get("question_polarity"),
                    "context_condition": row.get("context_condition"),
                    "category": row.get("category"),
                    "context": row.get("context", "").strip(),
                    "question": row.get("question", "").strip(),
                    "ans0": row.get("ans0", "").strip(),
                    "ans1": row.get("ans1", "").strip(),
                    "ans2": row.get("ans2", "").strip(),
                    "label": int(row.get("label")),
                    "answer_info": row.get("answer_info", {}),
                    "additional_metadata": row.get("additional_metadata", {}),
                    "choices": [
                        row.get("ans0", "").strip(),
                        row.get("ans1", "").strip(),
                        row.get("ans2", "").strip(),
                    ],
                }
                rows.append(ex)
    return Dataset.from_list(rows)


def load_test_datasets(root):
    all_lang_data = {}
    for lang in sorted(os.listdir(root)):
        lang_dir = os.path.join(root, lang)
        if not os.path.isdir(lang_dir):
            continue
        categories = {}
        for filename in sorted(os.listdir(lang_dir)):
            if filename.endswith(".jsonl"):
                fp = os.path.join(lang_dir, filename)
                categories[filename[:-6]] = load_single_test_file(fp)
        all_lang_data[lang] = categories
    return all_lang_data


def convert_testset_to_prompt_dataset(ds):
    rows = []
    for ex in ds:
        user_prompt = (
            f"Context: {ex['context']}\n\n"
            f"Question: {ex['question']}\n\n"
            f"Choices:\n"
            f"A: {ex['ans0']}\n"
            f"B: {ex['ans1']}\n"
            f"C: {ex['ans2']}\n\n"
            f"Answer with only a single letter: A, B, or C."
        )
        correct_answer = ["A", "B", "C"][ex["label"]]

        rows.append({
            "messages": [
                {"role": "system", "content": GLOBAL_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            "correct_answer": correct_answer,
            "category": ex["category"],
            "example_id": ex["example_id"],
        })
    return Dataset.from_list(rows)


# ============================================================
# Build task list
# ============================================================
tasks = []
for hardness in ["hard"]:#, "medium"]:
    root = hard_dict[hardness]
    test_data = load_test_datasets(root)
    for lang in sorted(test_data.keys()):
        for split in sorted(test_data[lang].keys()):
            tasks.append((hardness, lang, split, test_data))


# ============================================================
# MAIN LOOP
# ============================================================

for hardness, lang, split, test_data in tqdm(tasks, desc="All tasks"):

    print(f"\n[INFO] Hardness={hardness} | Lang={lang} | Split={split}")

    save_dir = f"/scratch/amukher6/cs795/results/{lang}/{hardness}/{model_version}/{think_folder}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{split}.json")

    ds = test_data[lang][split]
    prompt_ds = list(convert_testset_to_prompt_dataset(ds))

    # Skip logic
    if os.path.exists(save_path):
        try:
            with open(save_path, "r") as f:
                existing = json.load(f)
            expected_ids = {x["example_id"] for x in prompt_ds}
            existing_ids = {x["example_id"] for x in existing}
            if expected_ids == existing_ids:
                print(f"[SKIP] {save_path} already complete.")
                continue
        except:
            print("[WARN] Corrupt results → recomputing.")

    # Load model based on mode
    if mode == "baseline":
        model_path = MODEL_PATHS["baseline"]
    else:
        model_path = MODEL_PATHS[mode].format(lang=lang)

    print(f"[LOAD] Model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoLigerKernelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    all_results = []

    # ============================================================
    # BATCH INFERENCE LOOP
    # ============================================================

    total = len(prompt_ds)
    for start in tqdm(range(0, total, batch_size), desc=f"{lang}-{split}"):
        end = min(start + batch_size, total)
        batch = prompt_ds[start:end]

        prompts = [
            tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for ex in batch
        ]

        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        generated = model.generate(
            **model_inputs,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.0,
            max_new_tokens=num_think_tokens,
        )

        # batch decoding
        for i, ex in enumerate(batch):
            input_len = model_inputs.input_ids[i].size(0)
            gen_ids = generated[i][input_len:].tolist()

            # extract thinking
            try:
                cut = len(gen_ids) - gen_ids[::-1].index(151668)
            except ValueError:
                cut = 0

            thinking = tokenizer.decode(gen_ids[:cut], skip_special_tokens=True).strip()
            answer_section = tokenizer.decode(gen_ids[cut:], skip_special_tokens=True).strip()

            m = re.search(r"\b([ABC])\b", answer_section, re.I)
            final_answer = m.group(1).upper() if m else None

            all_results.append({
                "example_id": ex["example_id"],
                "lang": lang,
                "hardness": hardness,
                "split": split,
                "model": model_version,
                "final_answer": final_answer,
                "correct_answer": ex["correct_answer"],
                "is_correct": final_answer == ex["correct_answer"],
                "thinking": thinking,
                "raw_output": answer_section,
            })

    # Save
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"[✔] Saved → {save_path}")

print("\n[✔] Completed ALL inference runs.")
