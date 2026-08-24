import os
from typing import Dict, List

from datasets import load_dataset
from dotenv import load_dotenv

# Optional integrations
import wandb
from huggingface_hub import login

# Unsloth + TRL
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

# Pads input_ids with the pad token and labels with -100, keeping the
# prompt-masking below intact. (DataCollatorForLanguageModeling would
# rebuild labels from input_ids and silently undo the masking.)
from transformers import DataCollatorForSeq2Seq

from src.core.config import paths, training


class ChatDatasetWrapper:
    """
    Tokenizes {"messages": [system, user, assistant]} examples with the
    model's own chat template so training matches how Ollama serves the
    model, masks the prompt tokens in the labels, and guarantees an EOS
    token after the assistant JSON so generation stops after one object.
    """

    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _format_example(self, example: Dict) -> Dict:
        messages: List[Dict] = example["messages"]

        prompt_ids = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )

        if full_ids[-1] != self.tokenizer.eos_token_id:
            full_ids = full_ids + [self.tokenizer.eos_token_id]

        full_ids = full_ids[: self.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))

        # Loss only on the assistant JSON (and its EOS), never on the prompt.
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    def map(self):
        return self.dataset.map(self._format_example, remove_columns=self.dataset.column_names)


def run():
    load_dotenv()

    # Optional: HF + WandB login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        wandb.init(project="esco-json-eval", name="llama3.2-json-v2-chat")

    # -----------------------------
    # 1. Load model + tokenizer
    # -----------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=training.base_model_name,
        max_seq_length=training.max_seq_length,
        load_in_4bit=training.use_4bit_qlora,
    )

    # -----------------------------
    # 2. Apply LoRA
    # -----------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # -----------------------------
    # 3. Load dataset
    # -----------------------------
    hf_dataset = load_dataset(
        "json",
        data_files={
            "train": paths.train_file,
            "validation": paths.val_file,
        }
    )

    train_wrapper = ChatDatasetWrapper(
        hf_dataset=hf_dataset["train"],
        tokenizer=tokenizer,
        max_length=training.max_seq_length,
    )
    val_wrapper = ChatDatasetWrapper(
        hf_dataset=hf_dataset["validation"],
        tokenizer=tokenizer,
        max_length=training.max_seq_length,
    )

    tokenized_train = train_wrapper.map()
    tokenized_val = val_wrapper.map()

    # -----------------------------
    # 4. Data collator (keeps label masking)
    # -----------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # -----------------------------
    # 5. TRL SFT Trainer config
    # -----------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,  # REQUIRED
        args=SFTConfig(
            max_length=training.max_seq_length,
            dataset_num_proc=1,
            per_device_train_batch_size=training.batch_size,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            num_train_epochs=training.num_epochs,
            learning_rate=training.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            output_dir=training.output_dir,
            report_to="wandb" if wandb_api_key else "none",
        ),
    )

    # -----------------------------
    # 6. Train
    # -----------------------------
    trainer.train()

    # -----------------------------
    # 7. Save locally
    # -----------------------------
    trainer.save_model(training.output_dir)
    tokenizer.save_pretrained(training.output_dir)

    if wandb_api_key:
        wandb.finish()

    print("Training complete.")
