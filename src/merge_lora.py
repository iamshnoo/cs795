# # This code demonstrates loading a base model and merging it with LoRA adapters
# # from SFT/GRPO using PEFT.
# # ------------------------------------------------

# # tokenizer is same for all models
# # model1 = (base="Qwen/Qwen3-4B" + adapter=none)
# # model2 = (base=model1 + adapter=sft-adapter)
# # model3 = (base=model2 + adapter=grpo-adapter)
# # we will evaluate models 1, 2, and 3


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE_MODEL = "Qwen/Qwen3-4B"

def merge_model(mode, lang):
    """
    mode: "sft" or "sft_grpo"
    lang: english | hindi | french | german | chinese_simplified
    """
    # paths for adapters and outputs
    sft_adapter = f"/scratch/amukher6/cs795/train_out/{lang}_sft_lora"
    grpo_adapter = f"/scratch/amukher6/cs795/train_out/{lang}_sft_grpo_lora"

    model2_dir = f"/scratch/amukher6/cs795/models/{lang}_model2_sft"
    model3_dir = f"/scratch/amukher6/cs795/models/{lang}_model3_sft_grpo"
    os.makedirs(model2_dir, exist_ok=True)
    os.makedirs(model3_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if mode == "sft":
        print(f"[INFO] Creating model2 (base + SFT) for {lang}")

        # load model1 (base)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype="auto", device_map="auto"
        )

        # merge SFT → model2
        model_to_merge = PeftModel.from_pretrained(base_model, sft_adapter)
        merged_model = model_to_merge.merge_and_unload()
        merged_model.save_pretrained(model2_dir)
        tokenizer.save_pretrained(model2_dir)
        print(f"[✔] Saved model2 → {model2_dir}")

    elif mode == "sft_grpo":
        print(f"[INFO] Creating model3 (model2 + GRPO) for {lang}")

        # load model2
        base_model = AutoModelForCausalLM.from_pretrained(
            model2_dir, dtype="auto", device_map="auto"
        )

        # merge GRPO → model3
        model_to_merge = PeftModel.from_pretrained(base_model, grpo_adapter)
        merged_model = model_to_merge.merge_and_unload()
        merged_model.save_pretrained(model3_dir)
        tokenizer.save_pretrained(model3_dir)
        print(f"[✔] Saved model3 → {model3_dir}")

    else:
        raise ValueError("mode must be 'sft' or 'sft_grpo'")

if __name__ == "__main__":
    merge_model("sft_grpo", "hindi")
