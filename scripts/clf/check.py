import argparse
import random

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from checklist.editor import Editor
from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb

from util import read_jsonl, pipe_predict


def fix_yo(x, *args, **kwargs):
    if "ё" not in x:
        return None
    return x.replace("ё", "е")


def fix_exclamation(x, *args, **kwargs):
    if "!" not in x:
        return None
    return x.replace("!", "")


def fix_question(x, *args, **kwargs):
    if "?" not in x:
        return None
    return x.replace("?", "")


def fix_case(x, *args, **kwargs):
    has_lower = any(ch.islower() for ch in x)
    if has_lower:
        return None
    return x.lower()


class TextDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __getitem__(self, i):
        return self.records[i]["text"][:512]

    def __len__(self):
        return len(self.records)


def main(
    model_name,
    editor_model_name,
    test_path,
    sample_rate,
    seed
):
    random.seed(seed)
    editor = Editor(language="russian", model_name="xlm-roberta-large")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_num = 0 if device == "cuda" else -1

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)

    test_records = list(read_jsonl(test_path, sample_rate))
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=device_num
    )

    test_dataset = TextDataset(test_records)

    suite = TestSuite()
    suite.add(INV(
        **Perturb.perturb(test_dataset, fix_yo, keep_original=True),
        name="ё -> е",
        capability="Robustness",
        description=""
    ))
    suite.add(INV(
        **Perturb.perturb(test_dataset, fix_exclamation, keep_original=True),
        name="rm !",
        capability="Robustness",
        description=""
    ))
    suite.add(INV(
        **Perturb.perturb(test_dataset, fix_case, keep_original=True),
        name="CAPS -> lower",
        capability="Robustness",
        description=""
    ))
    suite.add(INV(
        **Perturb.perturb(test_dataset, fix_question, keep_original=True),
        name="rm ?",
        capability="Robustness",
        description=""
    ))

    suite.run(lambda x: pipe_predict(x, pipe), overwrite=True)
    suite.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--sample-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--editor-model-name", type=str, default="xlm-roberta-large")
    args = parser.parse_args()
    main(**vars(args))
