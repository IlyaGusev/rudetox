import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from util.dl import pipe_predict


def main(
    base_path,
    additional_path,
    clf_model_name,
    output_path
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_num = 0 if device == "cuda" else -1

    model = AutoModelForSequenceClassification.from_pretrained(clf_model_name)
    tokenizer = AutoTokenizer.from_pretrained(clf_model_name)
    model.to(device)

    base_texts = []
    with open(base_path) as r:
        for line in r:
            line = line.strip()
            base_texts.append(line)
    additional_texts = []
    with open(additional_path) as r:
        for line in r:
            line = line.strip()
            additional_texts.append(line)

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, framework="pt", device=device_num)
    y_pred, scores = pipe_predict(base_texts, pipe)
    for i, (label, text) in enumerate(zip(y_pred, additional_texts)):
        if label == 1:
            base_texts[i] = text
    with open(output_path, "w") as w:
        for text in base_texts:
            w.write(text.strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, required=True)
    parser.add_argument("--additional-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--clf-model-name", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
