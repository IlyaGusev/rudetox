import argparse
import random

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
# from checklist.editor import Editor
from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb

from util.io import read_jsonl
from util.dl import pipe_predict


def replace_yo(x):
    if "ё" not in x:
        return None
    return x.replace("ё", "е")


def rm_exclamation(x):
    if "!" not in x:
        return None
    return x.replace("!", "")


def add_exclamation(x):
    if "!" in x:
        return None
    if x[-1] == ".":
        return x[:-1] + "!"
    return x + "!"


def rm_question(x):
    if "?" not in x:
        return None
    return x.replace("?", "")


def fix_case(x):
    has_lower = any(ch.islower() for ch in x)
    if has_lower:
        return None
    return x.lower()


def concat_with_toxic(x, toxic_texts):
    toxic_comment = random.choice(toxic_texts)
    return " ".join((toxic_comment, x) if random.random() < 0.5 else (x, toxic_comment))


def concat_non_toxic(x, non_toxic_texts):
    non_toxic_comment = random.choice(non_toxic_texts)
    return " ".join((non_toxic_comment, x) if random.random() < 0.5 else (x, non_toxic_comment))


def add_toxic_words(x, toxic_words, num_words=3):
    sampled_words = []
    for _ in range(num_words):
        sampled_words.append(random.choice(toxic_words))
    return " ".join(sampled_words + [x])


class TextDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __getitem__(self, i):
        return self.records[i]["text"]

    def __len__(self):
        return len(self.records)


def main(
    model_name,
    editor_model_name,
    test_path,
    sample_rate,
    seed,
    toxic_vocab_path
):
    random.seed(seed)
    # editor = Editor(language="russian", model_name="xlm-roberta-large")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_num = 0 if device == "cuda" else -1

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)

    records = list(read_jsonl(test_path, sample_rate))
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=device_num
    )

    test_dataset = TextDataset(records)
    all_texts = [r["text"] for r in records]

    y_pred, scores = pipe_predict(all_texts, pipe)
    for r, p, score in zip(records, y_pred, scores):
        r["prediction"] = p
        r["score"] = score
    pred_zeros_texts = [r["text"] for r in records if r["prediction"] == 0]
    pred_ones_texts = [r["text"] for r in records if r["prediction"] == 1]

    suite = TestSuite()
    suite.add(INV(
        **Perturb.perturb(test_dataset, replace_yo, keep_original=True),
        name="ё -> е",
        capability="Robustness",
        description=""
    ))
    suite.add(INV(
        **Perturb.perturb(test_dataset, rm_exclamation, keep_original=True),
        name="rm !",
        capability="Robustness",
        description=""
    ))
    suite.add(INV(
        **Perturb.perturb(test_dataset, add_exclamation, keep_original=True),
        name="add !",
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
        **Perturb.perturb(test_dataset, rm_question, keep_original=True),
        name="rm ?",
        capability="Robustness",
        description=""
    ))
    suite.add(MFT(
        [concat_with_toxic(t, pred_ones_texts) for t in pred_zeros_texts],
        labels=1,
        name="Concat with non-toxic with toxic",
        capability="Logic",
        description=""
    ))
    suite.add(MFT(
        [concat_non_toxic(t, pred_zeros_texts) for t in pred_zeros_texts],
        labels=0,
        name="Concat with non-toxic with non-toxic",
        capability="Logic",
        description=""
    ))
    if toxic_vocab_path is not None:
        toxic_words = list()
        with open(toxic_vocab_path, "r") as r:
            for line in r:
                line = line.strip()
                toxic_words.append(line)
        suite.add(MFT(
            [add_toxic_words(t, toxic_words) for t in pred_zeros_texts],
            labels=1,
            name="Add toxic words",
            capability="Vocabulary",
            description=""
        ))

    suite.run(lambda x: pipe_predict(x, pipe), overwrite=True)
    suite.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--toxic-vocab-path", type=str, default=None)
    parser.add_argument("--sample-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--editor-model-name", type=str, default="xlm-roberta-large")
    args = parser.parse_args()
    main(**vars(args))
