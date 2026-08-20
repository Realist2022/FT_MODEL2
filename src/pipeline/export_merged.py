from unsloth import FastLanguageModel

checkpoint_path = "models/checkpoints/llama3.2-3b-esco-json"
repo_name = "Realist2022/cv-guestimator-llama3.2"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path,
    max_seq_length=2048,
    load_in_4bit=False,  # Float16 merge
)

# Merge LoRA adapter into the base model and push to HF
model.push_to_hub_merged(
    repo_name,
    tokenizer,
    save_method="merged_16bit",
)
print("Merged model pushed successfully!")