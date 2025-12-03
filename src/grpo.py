# This file demonstrates the GRPO training process using LoRA adapters with TRL.
# ------------------------------------------------

import argparse
import json
import os
import re
import wandb
import math
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig


def load_grpo_dataset(lang):
    """
    Load a GRPO dataset for one language in TRL message-style format.

    Returned dataset example:
    {
        "prompt": [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."}
        ],
        "chosen": [
            {"role": "assistant", "content": "<think>...</think><answer>...</answer>"}
        ],
        "rejected": [
            {"role": "assistant", "content": "<think>...</think><answer>...</answer>"}
        ]
    }
    """
    folder="/scratch/amukher6/cs795/grpo_data"
    path = os.path.join(folder, f"{lang}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found: {path}")

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            # Inline formatting of accept/reject with <think> + <answer>
            def format_output(text):
                text = text.strip()
                match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)

                if match:
                    reasoning = match.group(1).strip()
                    after = text.split("</think>", 1)[1].strip()
                    final = after
                else:
                    reasoning = ""
                    final = text

                return (
                    f"<think>\n{reasoning}\n</think>\n"
                    f"<answer>\n{final}\n</answer>"
                )

            # Build prompt as message list (system + user)
            prompt_messages = []
            if row.get("system", "").strip():
                prompt_messages.append({"role": "system", "content": row["system"].strip()})

            prompt_messages.append({"role": "user", "content": row.get("instruction", "").strip()})

            # Build chosen / rejected messages
            chosen_messages = [
                {"role": "assistant", "content": format_output(row.get("accept", ""))}
            ]

            rejected_messages = [
                {"role": "assistant", "content": format_output(row.get("reject", ""))}
            ]

            rows.append({
                "prompt": prompt_messages,
                "chosen": chosen_messages,
                "rejected": rejected_messages,
            })

    return Dataset.from_list(rows)

def format_reward(completions, **kwargs):
    rewards = []

    for comp in completions:
        # flatten if needed
        if isinstance(comp, list):
            comp = "".join([m.get("content", "") for m in comp])
        if not isinstance(comp, str):
            rewards.append(0.0)
            continue

        score = 0.0

        # 1. reward valid <think> ... </think> block
        think_match = re.search(r"<think>(.*?)</think>", comp, re.DOTALL)
        if think_match:
            score += 0.5

        # 2. reward correct order: <think>...</think> then answer letter
        #    pattern: </think> SOME_WHITESPACE A
        if re.search(r"</think>\s*[ABC]\b", comp):
            score += 0.25

        # 3. reward if final output ends with exactly ONE answer letter
        #    no trailing punctuation or explanation
        #    acceptable: "... </think>\n\nA"
        if re.search(r"[ABC]\s*$", comp):
            score += 0.25

        rewards.append(score)

    return rewards

def len_reward(completions, target_len=256, sigma=64, **kwargs):
    rewards = []

    for comp in completions:
        if isinstance(comp, list):
            comp = "".join([m.get("content", "") for m in comp])
        if not isinstance(comp, str):
            rewards.append(0.0)
            continue

        # extract think content
        m = re.search(r"<think>(.*?)</think>", comp, flags=re.DOTALL)
        if m:
            text = m.group(1)
        else:
            text = comp

        L = len(text.split())  # token-ish via whitespace

        # Gaussian centered on target_len
        # reward = exp(- (L - target_len)^2 / (2*sigma^2))
        reward = math.exp(-((L - target_len) ** 2) / (2 * sigma ** 2))

        rewards.append(reward)

    return rewards

if __name__ == "__main__":
    wandb.init(project="cs795", group="grpo")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="english", help="Language for GRPO training")
    args = parser.parse_args()
    lang = args.lang

    output_dir = f"/scratch/amukher6/cs795/train_out/{lang}_sft_grpo_lora"
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = load_grpo_dataset(lang)
    model_name = f"/scratch/amukher6/cs795/models/{lang}_model2_sft"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    peft_config = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.05, # default
        bias="none", # default
        target_modules=['v_proj', 'q_proj', 'gate_proj', 'k_proj', 'up_proj', 'down_proj', 'o_proj']
    )

    training_args = GRPOConfig(
        max_steps=250, #num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,

        max_completion_length=1024,
        max_prompt_length=2048,
        num_generations=4,
        generation_batch_size=8,

        output_dir=output_dir,
        logging_steps=1,
        log_completions=True,
        report_to="wandb",
        push_to_hub=False,
        use_liger_kernel=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, len_reward],
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    wandb.finish()
