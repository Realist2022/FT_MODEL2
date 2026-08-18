# src/train.py

from dataclasses import dataclass
from typing import Dict

from config import training, paths

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


@dataclass
class JsonExample:
    instruction: str
    output: Dict


class JsonDatasetWrapper:
    """
    Wraps HF dataset to return prompt + target text.
    """

    def __init__(self, hf_dataset, tokenizer, max_length: int):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _format_example(self, example: Dict) -> Dict:
        instruction = example["instruction"]
        output_json = example["output"]

        # Model sees target as plain JSON string
        target_str = json_dumps_compact(output_json)

        prompt = (
            "You are a strict JSON generator.\n"
            "Given the instruction, output ONLY the JSON object.\n\n"
            f"Instruction:\n{instruction}\n\n"
            "JSON output:\n"
        )

        text = prompt + target_str

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def map(self):
        return self.dataset.map(self._format_example, remove_columns=self.dataset.column_names)


def json_dumps_compact(obj: Dict) -> str:
    import json
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def main():
    # 1. Load tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(training.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        training.base_model_name,
        load_in_4bit=training.use_4bit_qlora,
        device_map="auto",
    )

    # 2. Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 3. Load dataset
    hf_dataset = load_dataset(
        "json",
        data_files={
            "train": paths.train_file,
            "validation": paths.val_file,
        }
    )

    train_wrapper = JsonDatasetWrapper(
        hf_dataset=hf_dataset["train"],
        tokenizer=tokenizer,
        max_length=training.max_seq_length,
    )
    val_wrapper = JsonDatasetWrapper(
        hf_dataset=hf_dataset["validation"],
        tokenizer=tokenizer,
        max_length=training.max_seq_length,
    )

    tokenized_train = train_wrapper.map()
    tokenized_val = val_wrapper.map()

    # 4. Training args
    args = TrainingArguments(
        output_dir=training.output_dir,
        num_train_epochs=training.num_epochs,
        per_device_train_batch_size=training.batch_size,
        per_device_eval_batch_size=training.batch_size,
        learning_rate=training.learning_rate,
        eval_strategy="epoch",
        logging_steps=50,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )

    trainer.train()
    trainer.save_model(training.output_dir)
    tokenizer.save_pretrained(training.output_dir)

    print("Training complete.")


if __name__ == "__main__":
    main()
