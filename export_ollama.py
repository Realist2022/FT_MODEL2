from unsloth import FastLanguageModel

# 1. Load your fine-tuned model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "models/checkpoints/llama3.2-3b-esco-json", # Or your local checkpoint folder
    max_seq_length = 8192,
    dtype = None,
    load_in_4bit = True,
)

# 2. Export to GGUF (this automatically merges the weights!)
print("Exporting to GGUF...")
model.save_pretrained_gguf(
    "cv-guestimator", 
    tokenizer, 
    quantization_method = "q4_k_m" # Q4_K_M is the industry standard for speed vs. accuracy
)
print("Export complete!")