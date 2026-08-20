import os
from dotenv import load_dotenv
from huggingface_hub import login
from unsloth import FastLanguageModel

# ⭐ Import your config
from src.core.config import training

load_dotenv()

def upload_lora():
    # Path where trainer.save_model() wrote your adapter
    adapter_path = training.output_dir  # e.g. "models/checkpoints/llama3.2-3b-esco-json"

    # Hugging Face repo
    repo_name = "Realist2026/cv-guestimator-llama3.2"

    # Token
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    print(f"🔷 Loading LoRA adapter from: {adapter_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=training.max_seq_length,
        load_in_4bit=True,
    )

    print(f"🔷 Uploading LoRA adapter to Hugging Face: {repo_name}")

    model.push_to_hub(repo_name, token=hf_token)
    tokenizer.push_to_hub(repo_name, token=hf_token)

    print("✅ Upload complete!")
    
if __name__ == "__main__":
    upload_lora()