import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments, pipeline
from tqdm import tqdm
from sklearn.metrics import classification_report

from rudetox.util.io import read_jsonl
from rudetox.util.dl import gen_batch


class LabeledTokensDataset(Dataset):
    def __init__(self, records, max_tokens, tokenizer, text_field, labels_field):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.records = list()
        for r in tqdm(records):
            inputs = self.embed_record(r[text_field])
            true_inputs = [i for i in inputs["input_ids"].tolist() if i != tokenizer.pad_token_id]
            labels = r[labels_field]
            assert len(true_inputs) == len(labels)
            labels = labels[:self.max_tokens] + [0 for _ in range(self.max_tokens - len(labels))]
            inputs["labels"] = labels
            assert len(inputs["input_ids"]) == len(labels)
            self.records.append(inputs)

    def embed_record(self, text):
        inputs = self.tokenizer(
            text=text,
            add_special_tokens=True,
            max_length=self.max_tokens,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt"
        )
        return {key: value.squeeze(0) for key, value in inputs.items()}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def main(
    train_path,
    val_path,
    config_path,
    seed,
    out_dir,
    sample_rate,
    text_field,
    labels_field
):
    train_records = list(read_jsonl(train_path, sample_rate))
    val_records = list(read_jsonl(val_path, sample_rate))

    with open(config_path, "r") as r:
        config = json.load(r)

    random.seed(seed)
    random.shuffle(train_records)
    print("Train records: ", len(train_records))
    print("Val records: ", len(val_records))

    max_tokens = config["max_tokens"]
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = LabeledTokensDataset(train_records, max_tokens, tokenizer, text_field, labels_field)
    val_dataset = LabeledTokensDataset(val_records, max_tokens, tokenizer, text_field, labels_field)

    num_labels = config["num_labels"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    model = model.to(device)
    model.config.id2label = {int(key): value for key, value in config["id2label"].items()}
    model.config.label2id = config["label2id"]

    batch_size = config["batch_size"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    logging_steps = config["logging_steps"]
    eval_steps = config["eval_steps"]
    save_steps = config["save_steps"]
    learning_rate = config["learning_rate"]
    warmup_steps = config["warmup_steps"]
    num_train_epochs = config["num_train_epochs"]
    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to="none",
        load_best_model_at_end=True,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    sample = "Горит восток зарёю новой! Говно, залупа, пенис, хер, давалка, хуй, блядина, хороший или плохой человек"
    model = model.to("cpu")
    logits = model(**tokenizer(sample, add_special_tokens=True, return_tensors="pt")).logits.squeeze(0)
    print(sample)
    print(torch.argmax(logits, dim=1).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--val-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--text-field", type=str, required=True)
    parser.add_argument("--labels-field", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
