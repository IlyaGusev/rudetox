import argparse
import json
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, pipeline
from tqdm import tqdm
from sklearn.metrics import classification_report

from rudetox.util.io import read_jsonl
from rudetox.util.dl import gen_batch


class LabeledDataset(Dataset):
    def __init__(
        self,
        records,
        max_tokens,
        tokenizer,
        text_field="text",
        res_field="label"
    ):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.records = list()
        for r in tqdm(records):
            inputs = self.embed_record(r[text_field])
            inputs["labels"] = torch.LongTensor([r[res_field]])
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


def pipe_predict(data, pipe, batch_size=64):
    raw_preds = []
    for batch in gen_batch(data, batch_size):
        raw_preds += pipe(batch)
    label2id = pipe.model.config.label2id
    y_pred = np.array([label2id[sample["label"]] for sample in raw_preds])
    scores = np.array([sample["score"] for sample in raw_preds])
    return y_pred, scores


def train(
    train_records,
    val_records,
    config,
    seed,
    text_field,
    res_field,
    device,
    output_dir=None
):
    random.seed(seed)
    random.shuffle(train_records)
    print("Train records: ", len(train_records))
    print("Val records: ", len(val_records))

    max_tokens = config["max_tokens"]
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    train_dataset = LabeledDataset(train_records, max_tokens, tokenizer, text_field, res_field)
    val_dataset = LabeledDataset(val_records, max_tokens, tokenizer, text_field, res_field)

    num_labels = config["num_labels"]
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
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
    if output_dir is None:
        temp_output_dir = tempfile.TemporaryDirectory()
        output_dir = temp_output_dir.name
    training_args = TrainingArguments(
        output_dir=output_dir,
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

    return model, tokenizer


def main(
    train_path,
    val_path,
    test_path,
    config_path,
    seed,
    out_dir,
    sample_rate,
    text_field,
    res_field,
    override_base_model
):
    train_records = list(read_jsonl(train_path, sample_rate))
    val_records = list(read_jsonl(val_path, sample_rate))
    test_records = list(read_jsonl(test_path, sample_rate))
    with open(config_path, "r") as r:
        config = json.load(r)
    if override_base_model:
        config["model_name"] = override_base_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = train(
        train_records,
        val_records,
        config=config,
        seed=seed,
        text_field=text_field,
        res_field=res_field,
        output_dir=out_dir,
        device=device
    )
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    model.eval()
    device_num = 0 if device == "cuda" else -1
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, framework="pt", device=device_num)
    y_pred, _ = pipe_predict([r["text"][:512] for r in test_records], pipe)
    y_true = [r["label"] for r in test_records]
    print(classification_report(y_true, y_pred, digits=3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--val-path", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--res-field", type=str, default="label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--override-base-model", type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))
