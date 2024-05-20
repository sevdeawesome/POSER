from transformers import AutoTokenizer, AutoModelForCausalLM
import fire
import os


def download_from_hf(model_id, output_dir, use_fast=True):
    if os.path.exists(f"{output_dir}/pytorch_model.bin"):
        print("Model already downloaded. Exiting...")
        return
    print("Downloading from HuggingFace...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=use_fast, trust_remote_code=True)

    tokenizer.save_pretrained(output_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True)

    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir, max_shard_size="20GB")

if __name__ == "__main__":
    fire.Fire(download_from_hf)