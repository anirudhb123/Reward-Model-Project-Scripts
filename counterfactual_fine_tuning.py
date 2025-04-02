#!/usr/bin/env python3
"""Fine-tune a preference model on new examples and push to Hugging Face Hub."""

import logging
import traceback
import time
import os
import sys

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fine_tuning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from absl import app
from absl import flags
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils import jsonl_utils
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
import random
import numpy as np
import evaluate
import tempfile
from torch.utils.data import DataLoader
from typing import Dict, Union, Any
import json
from transformers import BitsAndBytesConfig


_INPUT_PATH = flags.DEFINE_string(
    "input_path", "", "Path to the input file containing training examples."
)
_MODEL_REPO_ID = flags.DEFINE_string(
    "model_repo_id", "", "Hugging Face Hub repository ID (e.g., 'username/model-name')."
)
_HF_TOKEN = flags.DEFINE_string(
    "hf_token", "", "Hugging Face API token. If not provided, will look for HUGGING_FACE_HUB_TOKEN env var."
)
_BASE_MODEL_NAME = flags.DEFINE_string(
    "base_model_name", "Skywork/Skywork-Reward-Gemma-2-27B-v0.2", "Base model name."
)
_EPOCHS = flags.DEFINE_integer(
    "epochs", 3, "Number of training epochs."
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 8, "Training batch size."
)
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", 2e-5, "Learning rate for fine-tuning."
)
_USE_LORA = flags.DEFINE_boolean(
    "use_lora", True, "Whether to use LoRA for efficient fine-tuning."
)
_LORA_R = flags.DEFINE_integer(
    "lora_r", 16, "Rank for LoRA adapter."
)
_LORA_ALPHA = flags.DEFINE_integer(
    "lora_alpha", 32, "Alpha for LoRA adapter."
)
_VALIDATION_SPLIT = flags.DEFINE_float(
    "validation_split", 0.1, "Fraction of data to use for validation."
)
_SEED = flags.DEFINE_integer(
    "seed", 42, "Random seed for reproducibility."
)

def login_to_hub(token: str | None):
    if token:
        login(token=token, add_to_git_credential=False)
    else:
        print("❗ No HF token — skipping login")


class RewardTrainer:
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.best_accuracy = 0.0

    def train(self):
        print("🚀 Entering custom training method")
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        for epoch in range(self.args.num_train_epochs):
            print(f"🌟 Epoch {epoch+1}/{self.args.num_train_epochs}")
            self.model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                t0 = time.time()
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                torch.cuda.synchronize()

                torch.cuda.synchronize()
                t1 = time.time()
                outputs = self.model(**batch)
                scores = outputs.logits.squeeze(-1)
                loss = -torch.nn.functional.logsigmoid(scores[::2] - scores[1::2]).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                print(f"[Batch {batch_idx}] Compute: {time.time()-t1:.2f}s | Loss={loss.item():.4f}", flush=True)
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

            if self.eval_dataset:
                eval_results = self.evaluate()
                accuracy = eval_results["accuracy"]
                print(f"Validation Accuracy: {accuracy:.4f}")
                if accuracy > self.best_accuracy:
                    print(f"✅ New best accuracy ({accuracy:.4f} > {self.best_accuracy:.4f}), saving model locally")
                    self.best_accuracy = accuracy
                    self.save_model(save_to_hub=False)
                self.model.train()

    def evaluate(self):
        self.model.eval()
        loader = DataLoader(self.eval_dataset, batch_size=self.args.per_device_eval_batch_size, collate_fn=self.data_collator)
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                logits = self.model(**batch).logits.squeeze(-1)
                preds = logits.reshape(-1,2)[:,0] > logits.reshape(-1,2)[:,1]
                correct += preds.sum().item()
                total += preds.numel()
        return {"accuracy": correct/total}

    def save_model(self, save_to_hub=True):
        self.model.save_pretrained(self.args.output_dir)
        if save_to_hub:
            self.model.push_to_hub(self.args.hub_model_id)

def pairwise_data_collator(features: list) -> Dict[str, torch.Tensor]:
    """Collate paired examples (chosen vs rejected)"""
    batch = {
        "input_ids": [],
        "attention_mask": []
    }
    
    for feature in features:
        # Convert lists to tensors first
        batch["input_ids"].append(torch.tensor(feature["input_ids_chosen"]))
        batch["input_ids"].append(torch.tensor(feature["input_ids_rejected"]))
        batch["attention_mask"].append(torch.tensor(feature["attention_mask_chosen"]))
        batch["attention_mask"].append(torch.tensor(feature["attention_mask_rejected"]))
    
    return {
        "input_ids": torch.stack(batch["input_ids"]),
        "attention_mask": torch.stack(batch["attention_mask"]),
    }

def main(argv):    
    # Login to Hugging Face Hub
    login_to_hub(os.environ.get("HF_TOKEN"))
    
    print(f"Loading base model: {_BASE_MODEL_NAME.value}")
    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL_NAME.value)
    tokenizer.pad_token = tokenizer.eos_token  # Critical fix
    
    # Model loading with LoRA
    #bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_compute_dtype=torch.float16,
    #)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        _BASE_MODEL_NAME.value,
        quantization_config=bnb_config,
        device_map="cuda",
        num_labels=1,
    )
        
    if _USE_LORA.value:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=_LORA_R.value,
            lora_alpha=_LORA_ALPHA.value,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Prepare datasets
    train_dataset, val_dataset = prepare_counterfactual_data(
        _INPUT_PATH.value,
        tokenizer,
        val_split=_VALIDATION_SPLIT.value
    )

    # Training setup
    training_args = TrainingArguments(
        gradient_checkpointing=False, 
        output_dir=_MODEL_REPO_ID.value,  # Direct output to your repo
        learning_rate=_LEARNING_RATE.value,
        per_device_train_batch_size=_BATCH_SIZE.value,
        per_device_eval_batch_size=_BATCH_SIZE.value,
        num_train_epochs=_EPOCHS.value,
        evaluation_strategy="epoch",
        logging_steps=10,
        push_to_hub=True,  # Ensure this is enabled
        hub_model_id=_MODEL_REPO_ID.value,  # Your target repo
        hub_strategy="end",  # Push at end of training
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=pairwise_data_collator,  # Use custom collator
    compute_metrics=lambda preds: {
        "accuracy": (np.array(preds[0][::2]) > np.array(preds[0][1::2])).mean()
        }
    )
    
    print("Starting training...")
    trainer.train()
    print("🔖 Pushing best model to Hugging Face Hub…")
    trainer.save_model(save_to_hub=True)
    print("Done!")

def prepare_counterfactual_data(input_path, tokenizer, val_split=0.1):
    """Prepare dataset with correct pairing structure."""
    # Read and parse JSONL data
    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    random.shuffle(examples)
    
    # Split datasets
    split_idx = int(len(examples) * (1 - val_split))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    def process_batch(batch):
        # Process batch columns directly
        chosen = tokenizer.apply_chat_template(
            [
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": cr}
                ]
                for q, cr in zip(batch["query"], batch["chosen_response"])
            ],
            tokenize=False
        )
        
        rejected = tokenizer.apply_chat_template(
            [
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": rr}
                ]
                for q, rr in zip(batch["query"], batch["rejected_response"])
            ],
            tokenize=False
        )

        # Tokenize responses
        chosen_enc = tokenizer(
            chosen,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        rejected_enc = tokenizer(
            rejected,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        return {
            "input_ids_chosen": chosen_enc["input_ids"],  
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"],
        }

    # Create datasets with proper column structure
    train_dataset = Dataset.from_dict({
        "query": [ex["query"] for ex in train_examples],
        "chosen_response": [ex["chosen_response"] for ex in train_examples],
        "rejected_response": [ex["rejected_response"] for ex in train_examples]
    }).map(
        process_batch,
        batched=True,
        batch_size=4,
        remove_columns=["query", "chosen_response", "rejected_response"]
    )
    
    val_dataset = Dataset.from_dict({
        "query": [ex["query"] for ex in val_examples],
        "chosen_response": [ex["chosen_response"] for ex in val_examples],
        "rejected_response": [ex["rejected_response"] for ex in val_examples]
    }).map(
        process_batch,
        batched=True,
        batch_size=4,
        remove_columns=["query", "chosen_response", "rejected_response"]
    )
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    app.run(main)