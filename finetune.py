"""Fine-tune Mistral 7B on budget extraction using LoRA.

Requires: pip install unsloth datasets trl
Run on GPU (L4 24GB is sufficient).

Usage:
  python finetune.py                              # Default settings
  python finetune.py --epochs 3 --lr 2e-4         # Custom hyperparams
  python finetune.py --data training_data.jsonl    # Custom data file
"""

import argparse
import json

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


def load_training_data(path):
    """Load JSONL chat data into HF Dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return Dataset.from_list(examples)


def format_chat(example, tokenizer):
    """Apply chat template to messages."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training_data.jsonl")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--output", default="budget-mistral-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load and format data
    print(f"Loading training data: {args.data}")
    dataset = load_training_data(args.data)
    print(f"  {len(dataset)} examples")

    # Format with chat template
    dataset = dataset.map(
        lambda ex: {"text": format_chat(ex, tokenizer)},
        remove_columns=dataset.column_names,
    )

    # Train
    print(f"Training for {args.epochs} epochs...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=args.output,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            packing=False,
        ),
    )

    trainer.train()

    # Save LoRA adapter
    print(f"Saving LoRA adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Also save merged model for easy vLLM serving
    merged_dir = f"{args.output}-merged"
    print(f"Saving merged model to {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("Done! To serve with vLLM:")
    print(f"  vllm serve {merged_dir}")


if __name__ == "__main__":
    main()
