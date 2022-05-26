import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

from rudetox.util.io import read_jsonl
from rudetox.util.dl import pipe_predict


def main(
    model_name,
    test_path,
    sample_rate
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_num = 0 if device == "cuda" else -1

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)

    test_records = list(read_jsonl(test_path, sample_rate))
    test_records = [r for r in test_records if r["agreement"] > 0.7]
    text_records = list({r["text"]: r for r in test_records}.values())

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, framework="pt", device=device_num)
    y_pred, scores = pipe_predict([r["text"][:512] for r in test_records], pipe)
    y_true = [int(r["result"] == "toxic") for r in test_records]

    print("Errors:")
    print("Label 1:")
    for record, label, pred, score in zip(test_records, y_true, y_pred, scores):
        if label != pred and label == 1:
            print("Label: {}, prediction: {}, text: {}".format(label, pred, record["text"]))

    print("Label 0:")
    for record, label, pred, score in zip(test_records, y_true, y_pred, scores):
        if label != pred and label == 0:
            print("Label: {}, prediction: {}, text: {}".format(label, pred, record["text"]))

    print(classification_report(y_true, y_pred, digits=3))
    print("ROC AUC:", roc_auc_score(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
