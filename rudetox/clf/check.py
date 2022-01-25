import argparse
import os
import random

# For checklist
import nltk
nltk.download("omw-1.4")
nltk.download("wordnet")
nltk.download("wordnet_ic")
nltk.download("sentiwordnet")

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb

from rudetox.util.io import read_jsonl, read_lines
from rudetox.util.dl import pipe_predict
from rudetox.clf.transformations import replace_yo, rm_exclamation, add_exclamation, rm_question, fix_case
from rudetox.clf.transformations import concat_with_toxic, concat_non_toxic, add_toxic_words


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
    toxic_vocab_path,
    manual_test_path,
    save_path
):
    assert os.path.exists(test_path)
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_num = 0 if device == "cuda" else -1

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=True)
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

    inv_checks = [
        {
            "func": replace_yo,
            "name": "ё -> е",
            "capability": "Robustness",
            "description": ""
        },
        {
            "func": rm_exclamation,
            "name": "rm !",
            "capability": "Robustness",
            "description": ""
        },
        {
            "func": add_exclamation,
            "name": "add !",
            "capability": "Robustness",
            "description": ""
        },
        {
            "func": fix_case,
            "name": "CAPS -> lower",
            "capability": "Robustness",
            "description": ""
        },
        {
            "func": rm_question,
            "name": "rm ?",
            "capability": "Robustness",
            "description": ""
        }
    ]

    suite = TestSuite()
    for check in inv_checks:
        data = Perturb.perturb(test_dataset, check["func"], keep_original=True).data
        if not data:
            continue
        suite.add(INV(
            data=data,
            name=check["name"],
            capability=check["capability"],
            description=check["description"]
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
        toxic_words = read_lines(toxic_vocab_path)
        suite.add(MFT(
            [add_toxic_words(t, toxic_words) for t in pred_zeros_texts],
            labels=1,
            name="Add toxic words",
            capability="Vocabulary",
            description=""
        ))
    if manual_test_path is not None:
        manual_examples = list(read_jsonl(manual_test_path))
        manual_texts = [r["text"] for r in manual_examples]
        manual_labels = [r["label"] for r in manual_examples]
        assert len(manual_labels) == len(manual_texts)
        suite.add(MFT(
            manual_texts,
            labels=manual_labels,
            name="Manual examples",
            capability="Vocabulary",
            description=""
        ))

    suite.run(lambda x: pipe_predict(x, pipe), overwrite=True)
    if save_path:
        suite.save(save_path)
    suite.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--manual-test-path", type=str, default=None)
    parser.add_argument("--toxic-vocab-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--sample-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--editor-model-name", type=str, default="xlm-roberta-large")
    args = parser.parse_args()
    main(**vars(args))
