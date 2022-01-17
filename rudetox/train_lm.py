import argparse
import random
import json
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, logging
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from util.io import read_jsonl
from util.dl import set_random_seed, fix_tokenizer


class LMDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        sample_rate: float,
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        text_field: str = "text"
    ):
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count

        self.records = []
        for r in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert_text(r[text_field])
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def convert_text(self, text):
        input_ids = [self.tokenizer.bos_token_id]
        input_ids += self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_tokens_count,
            padding=True,
            truncation=True
        )["input_ids"]
        input_ids.append(self.tokenizer.eos_token_id)
        max_length = self.max_tokens_count
        padding = [self.tokenizer.pad_token_id for i in range(len(input_ids), max_length)]
        input_ids.extend(padding)
        input_ids = torch.LongTensor(input_ids)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


def train(
    config_file,
    train_file,
    val_file,
    sample_rate,
    output_dir,
    report_to,
    seed
):
    set_random_seed(seed)
    logging.set_verbosity_info()
    with open(config_file, "r") as r:
        config = json.load(r)

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    tokenizer = fix_tokenizer(tokenizer)

    train_records = list(read_jsonl(train_file))
    val_records = list(read_jsonl(val_file))
    random.shuffle(train_records)

    max_tokens_count = config["max_tokens_count"]
    train_dataset_args = {
        "original_records": train_records,
        "sample_rate": sample_rate,
        "tokenizer": tokenizer,
        "max_tokens_count": max_tokens_count
    }
    val_dataset_args = {
        "original_records": val_records,
        "sample_rate": sample_rate,
        "tokenizer": tokenizer,
        "max_tokens_count": max_tokens_count
    }
    train_dataset = LMDataset(**train_dataset_args)
    val_dataset = LMDataset(**val_dataset_args)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Special tokens
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model.config.pad_token_id is not None

    bos_candidates = (
        tokenizer.bos_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id
    )
    for bos_candidate in bos_candidates:
        model.config.bos_token_id = bos_candidate
        if bos_candidate is not None:
            break
    assert model.config.bos_token_id is not None
    model.config.decoder_start_token_id = model.config.bos_token_id

    eos_candidates = (tokenizer.eos_token_id, tokenizer.sep_token_id)
    for eos_candidate in eos_candidates:
        model.config.eos_token_id = eos_candidate
        if eos_candidate is not None:
            break
    assert model.config.eos_token_id is not None

    # Default model generation params
    model.config.num_beams = 5
    model.config.max_length = max_tokens_count

    # Training
    batch_size = config["batch_size"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    logging_steps = config["logging_steps"]
    eval_steps = config["eval_steps"]
    save_steps = config["save_steps"]
    learning_rate = config["learning_rate"]
    warmup_steps = config["warmup_steps"]
    num_train_epochs = config["num_train_epochs"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_steps=save_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", type=str, default="none")
    args = parser.parse_args()
    train(**vars(args))
