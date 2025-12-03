# This file demonstrates the SFT training process using LoRA adapters with TRL.
# ------------------------------------------------

from datasets import Dataset
import json
import os
import re

import wandb

wandb.init(project="cs795", group="sft")

LANGUAGE = "german"


def load_sft_dataset(lang, folder="/scratch/amukher6/cs795/sft_data"):
    """
    Load a single language JSONL file,
    convert it to TRL SFT-ready dataset, and enforce format:

        <think> ... </think>
        <answer> ... </answer>
    """

    path = os.path.join(folder, f"{lang}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found: {path}")

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            output = row.get("output", "").strip()

            # Try to extract reasoning
            think_match = re.search(r"<think>(.*?)</think>", output, flags=re.DOTALL)

            if think_match:
                reasoning = think_match.group(1).strip()
                # after </think>
                after = output.split("</think>", 1)[1].strip()
                final = after
            else:
                # No reasoning found → everything is answer
                reasoning = ""
                final = output

            formatted_output = (
                f"<think>\n{reasoning}\n</think>\n\n{final}"
                #f"<answer>\n{final}\n</answer>"
            )

            rows.append({
                "messages": [
                    {"role": "system",    "content": row.get("system", "")},
                    {"role": "user",      "content": row.get("instruction", "")},
                    {"role": "assistant", "content": formatted_output},
                ]
            })

    return Dataset.from_list(rows)

train_dataset = load_sft_dataset(LANGUAGE)
print(train_dataset)


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)


from transformers import Conv1D
def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []

    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            # model name parsing

            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])

    # Remove empty strings and 'lm-head' from the list
    layer_names = [name for name in layer_names if name and name != 'lm-head']
    return layer_names

print(list(set(get_specific_layer_names(model))))


from peft import LoraConfig
peft_config = LoraConfig(
    r=256,
    lora_alpha=16,
    lora_dropout=0.05, # default
    bias="none", # default
    target_modules=['v_proj', 'q_proj', 'gate_proj', 'k_proj', 'up_proj', 'down_proj', 'o_proj']
)


from trl import SFTConfig
import os


lang = LANGUAGE
output_dir = f"/scratch/amukher6/cs795/train_out/{lang}_sft_lora"
os.makedirs(output_dir, exist_ok=True)

training_args = SFTConfig(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    optim="adamw_bnb_8bit",
    max_length=None,
    output_dir=output_dir,
    logging_steps=1,
    report_to="wandb",
    push_to_hub=False,
    use_liger_kernel=True
)

from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    args=training_args,
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


trainer.save_model(output_dir)
# trainer.push_to_hub(dataset_name=dataset_name)
wandb.finish()
