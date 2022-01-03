import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import AutoTokenizer, pipeline

from util import read_jsonl, pipe_predict


def main(
    model_name,
    clf_name,
    input_path,
    output_path
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_num = 0 if device == "cuda" else -1

    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    records = read_jsonl(input_path)
    texts = [r["source"][:500] for r in records]
    pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=device_num
    )
    tokens_predictions, tokens_scores = pipe_predict(texts, pipe)

    clf_tokenizer = AutoTokenizer.from_pretrained(clf_name)
    clf_model = AutoModelForSequenceClassification.from_pretrained(clf_name).to(device)
    clf_pipe = pipeline(
        "text-classification",
        model=clf_model,
        tokenizer=clf_tokenizer,
        framework="pt",
        device=device_num
    )

    targets = []
    for text, predictions, scores in zip(texts, tokens_predictions, tokens_scores):
        tokens = tokenizer(text)["input_ids"]
        scores = scores[:len(tokens)]
        predictions = predictions[:len(tokens)]
        for i, (label, score) in enumerate(zip(predictions, scores)):
            if label == 0:
                scores[i] = 1.0 - scores[i]
        rm_indices = torch.argsort(torch.tensor(scores), descending=True)
        target = text
        iteration = 1
        while clf_pipe(target)[0]["label"] == "toxic":
            target_tokens = [token for i, token in enumerate(tokens) if i not in rm_indices[:iteration]]
            target = tokenizer.decode(target_tokens, skip_special_tokens=True)
            iteration += 1
        targets.append(target)

    with open(output_path, "w") as w:
        for t in targets:
            w.write(t.strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--clf-name", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
